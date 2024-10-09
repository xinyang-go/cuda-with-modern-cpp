cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_STANDARD 17)
project(hello_cuda LANGUAGES CXX CUDA)

add_compile_options(-Wall -Wextra)

add_executable(hello_cuda main.cu)