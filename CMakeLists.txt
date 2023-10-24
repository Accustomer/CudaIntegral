cmake_minimum_required(VERSION 2.6)

project(CudaIntegral)

set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_86;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_86;--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx512f -fopenmp")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/g++;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT;") 
  endif()
endif()

set(cuda_sources
  cpu_methods.cpp
  cpu_methods.h
  cuda_methods.cu
  cuda_methods.h
)  

set(sources
  compare.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cuda_integral ${cuda_sources} ${sources} OPTIONS -arch=sm_86)

set_target_properties(cuda_integral PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cuda_integral ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
