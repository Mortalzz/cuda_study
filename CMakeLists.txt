cmake_minimum_required(VERSION 3.28)

# Override with -DCMAKE_CUDA_ARCHITECTURES=native when changing GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(CudaLab LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Register each executable here; any additional sources follow the first one.
function(add_cuda_operator name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}/common")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    if(MSVC)
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endfunction()

add_cuda_operator(reduction 01_reduction/reduction.cu)
add_cuda_operator(transpose 00_transpose/transpose.cu)
add_cuda_operator(scan_prefix 02_scan_prefix/Hillis-Steele.cu)
