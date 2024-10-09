cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_parallel_for LANGUAGES CXX CUDA)

add_compile_options(-Wall -Wextra)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(cuda_parallel_for main.cu)