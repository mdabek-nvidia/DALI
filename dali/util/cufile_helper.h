// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_CUFILE_HELPER_H_
#define DALI_UTIL_CUFILE_HELPER_H_

// general stuff
#include <cstdio>
#include <string>
#include <utility>

#if !defined(__AARCH64_QNX__) && !defined(__AARCH64_GNU__) && !defined(__aarch64__)
#include <linux/sysctl.h>
#include <sys/syscall.h>
#endif
#include <unistd.h>
#include <fcntl.h>

// dali device guard
#include "dali/core/dynlink_cufile.h"
#include "dali/core/device_guard.h"
#include "dali/core/cuda_error.h"

namespace cufile {

struct CUFileDriverScope {
  CUFileDriverScope() {
    // v2 API performs proper reference counting, so we increase the reference count here...
    if (cuFileIsSymbolAvailable("cuFileDriverClose_v2")) {
      // cuFileDriverOpen in some versions of cuFile library, can close stdin
      // returning 0 file descriptor to the pool, then dali gets it from the OS opening a file
      // and passing to GDS which cannot handle it properly leading to an error
      int stdin_backup = dup(STDIN_FILENO);
      if (stdin_backup == -1) {
        std::cerr << "dup failed: " << strerror(errno) << "\n";
      }
      CUDA_CALL(cuFileDriverOpen());
      if (stdin_backup != -1) {
        if (fcntl(STDIN_FILENO, F_GETFL) == -1 && errno == EBADF) {
          // Restore stdin from backup
          if (dup2(stdin_backup, STDIN_FILENO) == -1) {
            std::cerr << "dup2 failed: " << strerror(errno) << "\n";
          }
        }
        close(stdin_backup);  // Cleanup backup
      }
    }
  }
  ~CUFileDriverScope() {
    // ...and decrease it here.
    // The old GDS API would simply destroy the library, possibly still in use by other modules
    // within the process.
    if (cuFileIsSymbolAvailable("cuFileDriverClose_v2")) {
      // cuFileDriverOpen in some versions of cuFile library, can close stdin
      // returning 0 file descriptor to the pool, then dali gets it from the OS opening a file
      // and passing to GDS which cannot handle it properly leading to an error
      int stdin_backup = dup(STDIN_FILENO);
      if (stdin_backup == -1) {
        std::cerr << "dup failed: " << strerror(errno) << "\n";
      }
      CUDA_DTOR_CALL(cuFileDriverClose());  // termination on exception is expected
      if (stdin_backup != -1) {
        if (fcntl(STDIN_FILENO, F_GETFL) == -1 && errno == EBADF) {
          // Restore stdin from backup
          if (dup2(stdin_backup, STDIN_FILENO) == -1) {
            std::cerr << "dup2 failed: " << strerror(errno) << "\n";
          }
        }
        close(stdin_backup);  // Cleanup backup
      }
    }
  }
};

// wrapper struct to conveniently store the fd's as well
class DLL_PUBLIC CUFileHandle {
 public:
  ~CUFileHandle() {
    Close();
  }
  CUFileHandle() = default;
  CUFileHandle(CUFileHandle &&other) {
    *this = std::move(other);
  }

  CUFileHandle &operator=(CUFileHandle &&other) {
    std::swap(fd, other.fd);
    std::swap(fdd, other.fdd);
    std::swap(cufh, other.cufh);
    other.Close();
    return *this;
  }

  void Close() {
    if (cufh) {
      cuFileHandleDeregister(cufh);
      cufh = nullptr;
    }
    if (fd != -1) close(fd);
    if (fdd != -1) close(fdd);
    fd = -1;
    fdd = -1;
  }

  CUfileHandle_t cufh = nullptr;
  int fd = -1;  // descriptor for buffered IO
  int fdd = -1;  // descriptor for direct IO
};

}  // namespace cufile

#endif  // DALI_UTIL_CUFILE_HELPER_H_
