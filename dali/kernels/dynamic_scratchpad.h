// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_DYNAMIC_SCRATCHPAD_H_
#define DALI_KERNELS_DYNAMIC_SCRATCHPAD_H_

#include <array>
#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>
#include "dali/core/static_switch.h"
#include "dali/core/mm/fixed_order_resource.h"
#include "dali/core/mm/memory.h"
#include "dali/core/mm/memory_kind.h"
#include "dali/core/mm/monotonic_resource.h"
#include "dali/kernels/context.h"
#include "dali/kernels/kernel_req.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename T, typename... Ts>
struct index_in_pack;

template <typename T, typename... Ts>
struct index_in_pack<T, T, Ts...> : std::integral_constant<int, 0> {};

template <typename T, typename U, typename... Ts>
struct index_in_pack<T, U, Ts...> :
    std::integral_constant<int, index_in_pack<T, Ts...>::value + 1> {};

/**
 * @brief Implements upstream handling and ordered wrappers.
 */
template <typename... Kinds>
class DynamicScratchpadImplT {
 protected:
  template <typename Kind>
  void set_upstream_resource(mm::memory_resource<Kind> *rsrc) {
    resource<Kind>() = mm::monotonic_memory_resource<Kind>(rsrc, initial_size<Kind>());
  }

  template <typename Kind>
  void set_upstream_resource(mm::async_memory_resource<Kind> *rsrc,
                             AccessOrder alloc_order,
                             AccessOrder dealloc_order = {}) {
    static_assert(!std::is_same<Kind, mm::memory_kind::host>::value,
      "Cannot use a stream-ordered resource for plain host memory");
    if (!dealloc_order.has_value())
      dealloc_order = alloc_order;
    adapter<Kind>() = { rsrc, alloc_order, dealloc_order };
    set_upstream_resource<Kind>(&adapter<Kind>());
  }

  template <typename Kind>
  size_t &initial_size() {
    return initial_sizes_[index_in_pack<Kind, Kinds...>::value];
  }

  template <typename Kind>
  size_t initial_size() const {
    return initial_sizes_[index_in_pack<Kind, Kinds...>::value];
  }

  template <typename Kind>
  mm::memory_resource<Kind> *get_upstream() const {
    return std::get<mm::monotonic_memory_resource<Kind>>(resources_)->get_upstream();
  }

  template <typename Kind>
  auto &adapter() {
    return std::get<mm::fixed_order_resource<Kind>>(adapters_);
  }

  template <typename Kind>
  auto &adapter() const {
    return std::get<mm::fixed_order_resource<Kind>>(adapters_);
  }

  template <typename Kind>
  auto &resource() {
    return std::get<mm::monotonic_memory_resource<Kind>>(resources_);
  }

  template <typename Kind>
  auto &resource() const {
    return std::get<mm::monotonic_memory_resource<Kind>>(resources_);
  }

  std::tuple<mm::fixed_order_resource<Kinds>...>      adapters_;
  std::tuple<mm::monotonic_memory_resource<Kinds>...> resources_;
  std::array<size_t, sizeof...(Kinds)>                initial_sizes_ = {};
};

using DynamicScratchpadImpl = DynamicScratchpadImplT<
      mm::memory_kind::host,
      mm::memory_kind::pinned,
      mm::memory_kind::device,
      mm::memory_kind::managed>;

}  // namespace detail

/**
 * @brief A dynamically allocated scratchpad
 *
 * A dynamic scratchpad dynamically allocates temporary buffers for each memory kind.
 * The memory used grows indefinitely and is freed once the object is destroyed (e.g. goes out
 * of scope). This means that instances of DynamicScratchpad MUST NOT be kept alive indefinitely,
 * e.g. as class members, because it will constitute an UNDETECTABLE functional MEMORY LEAK (the
 * buffers will be still reachable and will be freed when the scratchpad is destroyed, so memory
 * sanitizers won't complain).
 * Instead, a DynamicScratchpad should be declared as a local / temporary variable.
 *
 * The memory allocation and deallocation follows the specified access order (stream or host).
 * Device memory is allocated and deallocated in order specified in `device_order`.
 * Pinned memory is, by default, allocated in host order and deallocated in the same order as the
 * one used for device memory. These orders, however, can be specified explicitly.
 */
class DynamicScratchpad
  : public Scratchpad
  , private detail::DynamicScratchpadImpl {
 public:
  /**
   * @brief Constructs a dynamically allocated scratchpad
   *
   * @param device_order          Allocation and deallocation order for device memory.
   * @param pinned_dealloc_order  Deallocation order for pinned memory. Allocation is always
   *                              host-ordered. If not set, device_order is used.
   * @param managed_dealloc_order Deallocation order for managed memory. Allocation is always
   *                              host-ordered. If not set, device_order is used.
   * @param initial_sizes         Sizes, in bytes, of the initial buffers. Note that these buffers
   *                              are allocated lazily, so nothing is allocated if there's no request
   *                              for memory of any given kind.
   */
  using scratch_sizes_t = std::array<size_t, static_cast<size_t>(mm::memory_kind_id::count)>;
  explicit DynamicScratchpad(AccessOrder device_order = cudaStream_t(0),
                             AccessOrder pinned_dealloc_order = {},
                             AccessOrder managed_dealloc_order = {},
                             scratch_sizes_t initial_sizes = {}) {
    initial_sizes_ = initial_sizes;
    for (auto &s : initial_sizes_) {
      if (s == 0)
        s = 0x10000;  // 64k
    }
    if (!pinned_dealloc_order.has_value())
      pinned_dealloc_order = device_order;
    if (!managed_dealloc_order.has_value())
      managed_dealloc_order = device_order;

    device_order_ = device_order;
    pinned_dealloc_order_ = pinned_dealloc_order;
    managed_dealloc_order_ = managed_dealloc_order;
  }

  virtual void *Alloc(mm::memory_kind_id kind_id, size_t bytes, size_t alignment) {
    void *ret = nullptr;
    TYPE_SWITCH(kind_id, mm::kind2id, Kind,
      (mm::memory_kind::host,
       mm::memory_kind::pinned,
       mm::memory_kind::device,
       mm::memory_kind::managed),
      (ret = AllocImpl<Kind>(bytes, alignment)),
      (assert(!"Incorrect memory kind id");));
    return ret;
  }

  template <typename T>
  struct type_tag {};

  void InitResource(type_tag<mm::memory_kind::host>) {
    set_upstream_resource<mm::memory_kind::host>(mm::GetDefaultResource<mm::memory_kind::host>());
  }

  void InitResource(type_tag<mm::memory_kind::pinned>) {
    set_upstream_resource<mm::memory_kind::pinned>(
        mm::GetDefaultResource<mm::memory_kind::pinned>(),
        AccessOrder::host(),
        pinned_dealloc_order_);
  }

  void InitResource(type_tag<mm::memory_kind::device>) {
    set_upstream_resource<mm::memory_kind::device>(
        mm::GetDefaultResource<mm::memory_kind::device>(),
        device_order_);
  }

  void InitResource(type_tag<mm::memory_kind::managed>) {
    set_upstream_resource<mm::memory_kind::managed>(
        mm::GetDefaultResource<mm::memory_kind::managed>(),
        AccessOrder::host(),
        managed_dealloc_order_);
  }

  template <typename Kind>
  void *AllocImpl(size_t bytes, size_t alignment) {
    if (bytes == 0)
      return nullptr;  // do not initialize the resource in case of 0-sized allocation

    auto &r = resource<Kind>();
    if (!r.upstream()) {
      InitResource(type_tag<Kind>());
      assert(r.upstream() != nullptr);
    }
    return r.allocate(bytes, alignment);
  }

  AccessOrder device_order_, pinned_dealloc_order_, managed_dealloc_order_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_DYNAMIC_SCRATCHPAD_H_

