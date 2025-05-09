// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_INPUT_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_INPUT_OPERATOR_H_

#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/pipeline/operator/batch_size_provider.h"
#include "dali/pipeline/operator/builtin/caching_list.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/worker_thread.h"

namespace dali {

namespace detail {

template<typename Backend>
struct InputQueueItem {
  class EventLease {
    int device_id_ = -1;
    CUDAEvent event_;
   public:
    int device_id() const noexcept { return device_id_; }

    operator cudaEvent_t() const noexcept { return event_; }
    explicit operator bool() const noexcept { return event_; }

    void Get(int device_id) {
      if (device_id != device_id_)
        Put();
      if (!event_) {
        event_ = CUDAEventPool::instance().Get(device_id);
        device_id_ = device_id;
      }
    }

    void Put() {
      if (event_)
        CUDAEventPool::instance().Put(std::move(event_), device_id_);
      device_id_ = -1;
    }
  };

  TensorList<Backend> data;
  std::optional<std::string> data_id = std::nullopt;
  EventLease copy_complete;
  bool copy_performed = false;
  bool copy_requested = false;

  cudaEvent_t GetCompletionEvent(int device_id) {
    copy_complete.Get(device_id);
    return copy_complete;
  }
};

}  // namespace detail


/**
 * @brief Option used to override the InputOperator's copy mode, defined by ``no_copy`` parameter.
 *
 * It allows to:
 *  * DEFAULT - leave the default (the ``no_copy`` parameter is used),
 *  * FORCE_COPY - always make a copy,
 *  * FORCE_NO_COPY - always share the data without copy.
 */
enum class InputOperatorCopyMode {
  DEFAULT,
  FORCE_COPY,
  FORCE_NO_COPY
};


/**
 * @brief Options that can be configured when setting data for the External Source
 */
struct InputOperatorSettingMode {
  /**
   * @brief If SetExternalInputHelper should be blocking - waits until provided data is copied
   *        to the internal buffer
   */
  bool sync = false;

  /**
   * @brief If true, a copy kernel will be used to make a contiguous buffer instead of
   *  cudaMemcpyAsync.
   */
  bool use_copy_kernel = false;

  /**
   * @brief Select whether to use the parameter defined in the External Source or
   *  override the mode of operation forcing the copy or no-copy
   */
  InputOperatorCopyMode copy_mode = InputOperatorCopyMode::DEFAULT;
};


/**
 * InputOperator is an Operator that serves a purpose of in-memory input to a DALI Pipeline.
 * It has no regular inputs, but provides any number of output. The special feature of an
 * InputOperator is the CachingList - queue of input data.
 *
 * InputOperator API consists of three main parts:
 * 1. SetDataSource          - this is a set of functions, that enqueues the data in the Operator.
 *                             User shall call one of those functions prior to Operator::Run().
 * 2. ForwardCurrentData     - This function is used to retrieve the data that exists on the
 *                             top of the Operators queue.
 * 3. HandleDataAvailability - This function handles `blocking_` Operator parameter. Any subclass
 *                             of InputOperator shall call this function at the beginning of
 *                             its SetupImpl.
 */
template<typename Backend>
class InputOperator : public Operator<Backend>, virtual public BatchSizeProvider {
  using InBackend = std::conditional_t<
          std::is_same_v<Backend, GPUBackend>,
          GPUBackend /* GPUBackend */,
          CPUBackend /* CPUBackend and MixedBackend */>;
  using OutBackend = std::conditional_t<
          std::is_same_v<Backend, CPUBackend>,
          CPUBackend /* CPUBackend */,
          GPUBackend /* GPUBackend and MixedBackend */>;
  using InputQueue = CachingList<detail::InputQueueItem<InBackend>>;
  using queue_item_t = typename InputQueue::Item;

 public:
  explicit InputOperator(const OpSpec &spec) :
          Operator<Backend>(spec),
          device_id_(spec.GetArgument<int>("device_id")),
          blocking_(spec.GetArgument<bool>("blocking")),
          no_copy_(spec.GetArgument<bool>("no_copy")),
          sync_worker_(device_id_, false, "InputOperator sync_worker_") {
    if (std::is_same<Backend, GPUBackend>::value) {
      internal_copy_stream_ = CUDAStreamPool::instance().Get(device_id_);
      internal_copy_order_ = internal_copy_stream_;
    }
    sync_worker_.WaitForInit();
  }


  virtual ~InputOperator() {
    sync_worker_.ForceStop();
    sync_worker_.Shutdown();
  }

  DISABLE_COPY_MOVE_ASSIGN(InputOperator);


  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   *
   * @param data_id Arbitrary ID of the data passed to the function. Can be any string.
   */
  template<typename SrcBackend>
  void SetDataSource(const vector<Tensor<SrcBackend>> &vect_of_tensors, AccessOrder order = {},
                     InputOperatorSettingMode ext_src_setting_mode = {},
                     std::optional<std::string> data_id = std::nullopt) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][InputOperator] SetDataSource", DomainTimeRange::kViolet);
    DALI_ENFORCE(vect_of_tensors.size() > 0, "Provided batch cannot be empty.");
    TensorList<SrcBackend> tl(vect_of_tensors.size());
    tl.SetupLike(vect_of_tensors[0]);
    for (int i = 0; i < tl.num_samples(); ++i) {
      tl.SetSample(i, const_cast<Tensor<SrcBackend> &>(vect_of_tensors[i]));
    }
    SetDataSourceHelper(tl, std::move(data_id), order, ext_src_setting_mode);
  }


  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   *
   * @param data_id Arbitrary ID of the data passed to the function. Can be any string.
   */
  template<typename SrcBackend>
  void SetDataSource(const TensorList<SrcBackend> &tl, AccessOrder order = {},
                     InputOperatorSettingMode ext_src_setting_mode = {},
                     std::optional<std::string> data_id = std::nullopt) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][InputOperator] SetDataSource", DomainTimeRange::kViolet);
    SetDataSourceHelper(tl, std::move(data_id), order, ext_src_setting_mode);
  }

  /**
   * Returns the layout at the input of this Operator.
   */
  virtual const TensorLayout& in_layout() const = 0;

  /**
   * Returns the number of dimensions at the input of this Operator.
   */
  virtual int in_ndim() const = 0;

  /**
   * Returns the type of the data at the input of this Operator.
   */
  virtual DALIDataType in_dtype() const = 0;

  bool WouldCopy(InputOperatorCopyMode mode) const {
    switch (mode) {
      case InputOperatorCopyMode::FORCE_COPY:
        return true;
      case InputOperatorCopyMode::FORCE_NO_COPY:
        return false;
      default:
        return !no_copy_;
    }
  }

  /**
   * Break waiting for the next batch of data
   */
  void BreakWaiting() {
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      running_ = false;
    }
    cv_.notify_all();
  }


 protected:
  /**
   * Checks if there is more data in queue to be loaded.
   */
  bool HasDataInQueue() const {
    return !tl_data_.IsEmpty();
  }


  /**
   * Checks if the data is available. If it's not, either blocks or throws an error,
   * depending on ``blocking`` operator argument.
   *
   * Any Operator that inherits from InputOperator and uses ``blocking`` feature
   * shall call this function at the beginning of its SetupImpl.
   */
  void HandleDataAvailability() {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    if (blocking_) {
      cv_.wait(busy_lock, [&] { return HasDataInQueue(); });
    } else {
      if (!HasDataInQueue()) {
        DALI_FAIL("No data was provided to the InputOperator. Make sure to feed it properly.");
      }
    }
  }


  ///@{
  /**
   * Injects current data portion into the provided TensorList and recycles
   * the inner container for the data.
   *
   * This function will take a best effort not to copy the data,
   * however it might not always be possible.
   *
   * @param target Where the data shall be injected.
   * @param target_data_id Where the ID of the current data shall be injected.
   *                       @see named_pointer_to_tensor_list_t.
   * @param tp TheadPool used to copy the data.
   */
  void DLL_PUBLIC
  ForwardCurrentData(TensorList<CPUBackend> &target, std::optional<std::string> &target_data_id,
                     ThreadPool &tp);

  void DLL_PUBLIC
  ForwardCurrentData(TensorList<GPUBackend> &target, std::optional<std::string> &target_data_id,
                     cudaStream_t stream = nullptr);
  ///@}


  /**
   * Peeks the data that is next in line.
   */
  const TensorList<InBackend> &PeekCurrentData() {
    return tl_data_.PeekFront().data;
  }


  int NextBatchSize() override {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    if (blocking_) {
      cv_.wait(busy_lock, [&data = tl_data_, &running = running_] {
                             return !running || data.CanProphetAdvance();
                           });
    }
    return tl_data_.PeekProphet().data.num_samples();
  }


  void Advance() override {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    if (blocking_) {
      cv_.wait(busy_lock, [&data = tl_data_, &running = running_] {
                             return !running || data.CanProphetAdvance();
                           });
    }
    tl_data_.AdvanceProphet();
  }

  /**
   * "depleted" operator trace specifies whether the operator has sufficient resources to
   * run another iteration.
   *
   * If "false", the operator needs to be fed with data to run the next iteration. If "true",
   * the next iteration can be triggered.
   * @param ws Current workspace.
   * @param depleted Value of the trace.
   */
  void SetDepletedOperatorTrace(Workspace& ws, bool depleted) {
    ws.SetOperatorTrace("depleted", depleted ? "true" : "false");
  }


  int device_id_ = -1;
  bool blocking_ = true;
  bool no_copy_ = false;
  bool running_ = true;


 private:
  void RecycleBuffer(queue_item_t &&data) {
    data->copy_complete.Put();
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    tl_data_.Recycle(std::move(data));
  }


  template<typename SrcBackend>
  std::enable_if_t<!std::is_same<SrcBackend, InBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &t, std::optional<std::string> /* data_id */,
                AccessOrder /* order = {}*/, bool /* use_copy_kernel */) {
    DALI_FAIL(make_string("no_copy is supported only for the same data source device type "
                          "as operator. Received: ",
                          std::is_same<SrcBackend, CPUBackend>::value ? "CPU" : "GPU",
                          " input for ",
                          std::is_same<Backend, CPUBackend>::value ? "CPU" : "GPU",
                          " operator."));
  }


  template<typename SrcBackend>
  std::enable_if_t<
          std::is_same<SrcBackend, InBackend>::value && std::is_same<SrcBackend, CPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, std::optional<std::string> data_id,
                AccessOrder /* order = {}*/, bool /*use_copy_kernel = false*/) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    auto tl_elm = GetEmptyOutputBatch(std::move(data_id));
    tl_elm->copy_requested = false;
    tl_elm->copy_performed = true;
    // set pinned if needed
    if (batch.is_pinned() != tl_elm->data.is_pinned()) {
      tl_elm->data.Reset();
      tl_elm->data.set_pinned(batch.is_pinned());
    }
    tl_elm->data.ShareData(const_cast<TensorList<CPUBackend> &>(batch));
    tl_data_.PushBack(std::move(tl_elm));
  }


  /**
   * @brief Attempts to share data from tensor vector to tensor list without
   *        an additional copy if the batch is contiguous.
   *        In case of scattered samples, the data is copied to a contiguous
   *        buffer.
   * @remarks Mixing contiguous and non-contiguous inputs in subsequents calls
   *        is not supported and could lead to data corruption.
   * @param batch source data
   * @param data_id Arbitrary ID of the data passed to the function. Can be any string.
   * @param order CUDA stream use to schedule the copy (or host order to make the copy
   *              host-syncrhonous)
   * @param use_copy_kernel If true, a copy kernel will be used to make a
   *        contiguous buffer instead of cudaMemcpyAsync.
   */
  template<typename SrcBackend>
  std::enable_if_t<
          std::is_same<SrcBackend, InBackend>::value && std::is_same<SrcBackend, GPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, std::optional<std::string> data_id,
                AccessOrder order = {}, bool use_copy_kernel = false) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    auto tl_elm = GetEmptyOutputBatch(std::move(data_id));
    bool copied_shared_data = false;

    if (!order.has_value())
      order = batch.order().is_device() ? batch.order() : tl_elm->data.order();

    // We can share only contiguous tensor lists that are stored on the same device.
    if (batch.IsContiguousInMemory() && batch.device_id() == device_id_) {
      tl_elm->data.ShareData(batch);
      zero_copy_noncontiguous_gpu_input_ = true;
    } else {
      // Do not overwrite the buffer it if shares data.
      if (tl_elm->data.shares_data())
        tl_elm->data.Reset();
      tl_elm->data.Copy(batch, order, use_copy_kernel);

      if (order.is_device()) {
        cudaEvent_t event = tl_elm->GetCompletionEvent(order.device_id());
        DeviceGuard dg(order.device_id());
        CUDA_CALL(cudaEventRecord(event, order.stream()));
      }

      if (zero_copy_noncontiguous_gpu_input_) {
        DALI_WARN("ExternalSource operator should not mix contiguous and noncontiguous inputs. "
                  "In such a case the internal memory used to gather data in a contiguous chunk "
                  "of memory would be trashed.");
      }
      copied_shared_data = true;
    }
    tl_elm->copy_performed = copied_shared_data;
    tl_elm->copy_requested = false;
    tl_data_.PushBack(std::move(tl_elm));
  }


  template<typename SrcBackend, typename B = InBackend>
  std::enable_if_t<std::is_same<B, CPUBackend>::value>
  CopyUserData(const TensorList<SrcBackend> &batch, std::optional<std::string> data_id,
               AccessOrder order, bool /* sync */, bool /* use_copy_kernel */) {
    queue_item_t tl_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm = GetEmptyOutputBatch(std::move(data_id));
    }
    // set pinned if needed
    tl_elm->data.set_order(AccessOrder::host());
    if (batch.is_pinned() != tl_elm->data.is_pinned()) {
      tl_elm->data.Reset();
      tl_elm->data.set_pinned(batch.is_pinned());
      if constexpr (std::is_same_v<Backend, CPUBackend>)
        tl_elm->data.set_device_id(tl_elm->data.is_pinned() ? device_id_ : CPU_ONLY_DEVICE_ID);
    }
    AccessOrder copy_order =
            std::is_same<SrcBackend, CPUBackend>::value
            ? AccessOrder::host()  // do not use a device order for a host to host copy
            : order;
    tl_elm->data.Copy(batch, copy_order);
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm->copy_requested = true;
      tl_elm->copy_performed = true;
      tl_data_.PushBack(std::move(tl_elm));
    }
  }


  template<typename SrcBackend, typename B = InBackend>
  std::enable_if_t<std::is_same<B, GPUBackend>::value>
  CopyUserData(const TensorList<SrcBackend> &batch, std::optional<std::string> data_id,
               AccessOrder order, bool sync, bool use_copy_kernel) {
    queue_item_t tl_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm = GetEmptyOutputBatch(std::move(data_id));
    }
    // If we got a host order we most probably got it via FeedPipeline and we are trying to pass the
    // data from CPU to GPU. As we keep the order in tl_data_ as internal_copy_stream_, we will use
    // an actual stream for running and synchronizing with the copy. Note that the Copy can be truly
    // asynchronous if it comes from pinned memory or happens on a device with integrated memory
    // (like Xavier) where CPU and GPU share the same memory.
    if (!order.is_device()) {
      order = tl_elm->data.order();
    }
    tl_elm->data.Copy(batch, order, use_copy_kernel);
    int copy_device = order.is_device() ? order.device_id() : tl_elm->data.device_id();

    {
      DeviceGuard dg(copy_device);
      auto event = tl_elm->GetCompletionEvent(copy_device);
      CUDA_CALL(cudaEventRecord(event, order.stream()));
      if (sync) {
        CUDA_CALL(cudaEventSynchronize(event));
      }
    }

    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm->copy_requested = true;
      tl_elm->copy_performed = true;
      tl_data_.PushBack(std::move(tl_elm));
    }
  }

  template<typename SrcBackend>
  void SetDataSourceHelper(const TensorList<SrcBackend> &batch, std::optional<std::string> data_id,
                           AccessOrder order = {},
                           InputOperatorSettingMode ext_src_setting_mode = {}) {
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.

    bool actual_no_copy = no_copy_;
    switch (ext_src_setting_mode.copy_mode) {
      case InputOperatorCopyMode::FORCE_COPY:
        actual_no_copy = false;
        break;
      case InputOperatorCopyMode::FORCE_NO_COPY:
        actual_no_copy = true;
        break;
      default:
        actual_no_copy = no_copy_;
        break;
    }

    if (actual_no_copy) {
      ShareUserData(batch, std::move(data_id), order, ext_src_setting_mode.use_copy_kernel);
    } else {
      CopyUserData(batch, std::move(data_id), order, ext_src_setting_mode.sync,
                   ext_src_setting_mode.use_copy_kernel);
    }
    cv_.notify_one();
  }


  /**
   * @brief Get the empty output batch from tl_data_, first assigning the correct order to it.
   * @warning User is responsible for holding busy_m_ mutex when calling this function.
   *
   * @param data_id Arbitrary ID of the data. Can be any string.
   */
  queue_item_t GetEmptyOutputBatch(std::optional<std::string> data_id) {
    auto result = tl_data_.GetEmpty();
    int data_device_id = std::is_same_v<Backend, GPUBackend> || result->data.is_pinned()
      ? device_id_ : CPU_ONLY_DEVICE_ID;
    result->data.set_device_id(data_device_id);
    result->data.set_order(internal_copy_order_);
    result->data_id = (std::move(data_id));
    return result;
  }


  InputQueue tl_data_;

  std::mutex busy_m_;
  std::condition_variable cv_;

  /*
   * indicates that user provide noncontiguous GPU input with zero copy option so DALI needs
   * to create an internal copy, it is used to raise a warning when the user mixes contiguous and
   * noncontiguous GPU inputs with zero copy what trashed GPU allocated memory
   */
  bool zero_copy_noncontiguous_gpu_input_ = false;

  WorkerThread sync_worker_;
  CUDAStreamLease internal_copy_stream_ = {};
  AccessOrder internal_copy_order_ = AccessOrder::host();
};

/** Checks if the Operator is an InputOperator */
inline bool IsInputOperator(OperatorBase *op) {
  return dynamic_cast<InputOperator<CPUBackend> *>(op) ||
         dynamic_cast<InputOperator<MixedBackend> *>(op) ||
         dynamic_cast<InputOperator<GPUBackend> *>(op);
}

/** Checks, if the Operator defined by provided Schema is an InputOperator */
inline bool IsInputOperator(const OpSchema &schema) {
  const auto &parents = schema.GetParentNames();
  return std::any_of(parents.begin(), parents.end(),
                     [](const std::string &p) { return p == "InputOperatorBase"; });
}


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_INPUT_OPERATOR_H_
