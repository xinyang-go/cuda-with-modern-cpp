#pragma once
#ifndef CUDA_ALLOCATOR_HPP
#define CUDA_ALLOCATOR_HPP

#include <cuda.h>
#include <cstdint>

template <typename T>
struct cuda_alloctor {
    using value_type = T;
    using pointer = T *;

    pointer allocate(std::size_t n) {
        pointer ptr = nullptr;
        cudaMallocManaged(&ptr, n * sizeof(value_type));
        return ptr;
    }

    void deallocate(pointer ptr, [[maybe_unused]] std::size_t n) { cudaFree(ptr); }
};

#endif // CUDA_ALLOCATOR_HPP