cmake_minimum_required(VERSION 3.13)
project(CustomCppLib)
find_package(CUDA REQUIRED)
include(/opt/dali/internal_tools/find_dali.cmake)
add_library(test_lib SHARED main_stub.cc)
find_dali(DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)
target_include_directories(test_lib PUBLIC ${DALI_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_directories(test_lib PUBLIC ${DALI_LIB_DIR})
target_link_libraries(test_lib ${DALI_LIBRARIES} ${CUDA_LIBRARIES})
