// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_C_API_2_PIPELINE_H_
#define DALI_C_API_2_PIPELINE_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "dali/dali.h"
#include "dali/c_api_2/checkpoint.h"
#include "dali/c_api_2/pipeline_outputs.h"

// A dummy base that the handle points to
struct _DALIPipeline {
 protected:
  _DALIPipeline() = default;
  ~_DALIPipeline() = default;
};

namespace dali::c_api {

class PipelineWrapper : public _DALIPipeline {
 public:
  explicit PipelineWrapper(const daliPipelineParams_t &params);
  PipelineWrapper(const void *serialized, size_t length, const daliPipelineParams_t &params);
  ~PipelineWrapper();

  std::unique_ptr<PipelineOutputs> PopOutputs(AccessOrder order = AccessOrder::host());

  void Build();

  void Run();

  void Prefetch();

  int GetFeedCount(std::string_view input_name);

  void FeedInput(
      std::string_view input_name,
      const ITensorList *input_data,
      std::optional<std::string_view> data_id,
      daliFeedInputFlags_t options,
      AccessOrder order);

  int GetOutputCount() const;

  daliPipelineIODesc_t GetOutputDesc(int idx) const &;

  int GetInputCount() const;

  daliPipelineIODesc_t GetInputDesc(int idx) const &;

  daliPipelineIODesc_t GetInputDesc(std::string_view name) const &;

  /** Retrieves the underlying DALI Pipeline object */
  dali::Pipeline *Unwrap() const & {
    return pipeline_.get();
  }

  std::unique_ptr<CheckpointWrapper> GetCheckpoint(const daliCheckpointExternalData_t *ext) const;

  std::string_view SerializeCheckpoint(CheckpointWrapper &chk) const;

  std::unique_ptr<CheckpointWrapper> DeserializeCheckpoint(std::string_view serialized);

  void RestoreFromCheckpoint(CheckpointWrapper &chk);


 private:
  template <typename Backend>
  void FeedInputImpl(
        std::string_view input_name,
        const TensorList<Backend> &tl,
        std::optional<std::string_view> data_id,
        daliFeedInputFlags_t options,
        AccessOrder order);

  std::unique_ptr<Pipeline> pipeline_;
  mutable std::vector<std::string_view> input_names_;
};

PipelineWrapper *ToPointer(daliPipeline_h handle);
CheckpointWrapper *ToPointer(daliCheckpoint_h handle);

}  // namespace dali::c_api

#endif  // DALI_C_API_2_PIPELINE_H_
