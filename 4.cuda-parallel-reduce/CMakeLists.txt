cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_parallel_reduce LANGUAGES CXX CUDA)

add_compile_options(-Wall -Wextra)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Release)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--dopt=on>)
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--dopt=on>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
elseif(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(cuda_parallel_reduce main.cu)