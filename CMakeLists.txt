cmake_minimum_required(VERSION 3.24)
project(cuda_with_modern_cpp)
file(GLOB dirs [0-9]*.*)
foreach(dir ${dirs})
    add_subdirectory(${dir})
endforeach()
