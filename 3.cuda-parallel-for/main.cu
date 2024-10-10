#include <iostream>
#include <numeric>
#include <vector>

#include "../2.cuda-allocator/cuda_allocator.hpp"

#define CheckCudaApi(...)                                                                                \
    do {                                                                                                 \
        cudaError_t err = (__VA_ARGS__);                                                                 \
        if (err != cudaSuccess) {                                                                        \
            std::cerr << "Error: '" #__VA_ARGS__ "' = (" << cudaGetErrorString(err) << ")" << std::endl; \
            std::terminate();                                                                            \
        }                                                                                                \
    } while (0)

#define CheckCudaKernel(...)                                                                             \
    do {                                                                                                 \
        (__VA_ARGS__);                                                                                   \
        cudaError_t err = cudaGetLastError();                                                            \
        if (err != cudaSuccess) {                                                                        \
            std::cerr << "Error: '" #__VA_ARGS__ "' = (" << cudaGetErrorString(err) << ")" << std::endl; \
            std::terminate();                                                                            \
        }                                                                                                \
    } while (0)

template <typename T>
using cuda_vector = std::vector<T, cuda_alloctor<T>>;

// NOTICE: argument 'op' cannot be a reference, because it will reference to cpu memory and cause runtime error.
template <typename F>
void __global__ cuda_parallel_for_kernel(int n, F op) {
    int idx = blockIdx.x * gridDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += step) {
        op(i);
    }
}

template <typename F>
void cuda_parallel_for(int n, F &&op, int gridDim = 1, int blockDim = 128) {
    CheckCudaKernel(cuda_parallel_for_kernel<<<gridDim, blockDim>>>(n, op));
    CheckCudaApi(cudaDeviceSynchronize());
}

template <typename T, typename A>
std::ostream &operator<<(std::ostream &os, const std::vector<T, A> &vec) {
    if (vec.empty()) return os << "[]";
    os << "[" << vec[0];
    for (size_t i = 1; i < vec.size(); i++) os << ", " << vec[i];
    return os << "]";
}

void two_vector_add() {
    cuda_vector<int> a{0, 1, 2, 3};
    cuda_vector<int> b{4, 5, 6, 7};
    cuda_vector<int> c(4);

    cuda_parallel_for(a.size(), [a = a.data(), b = b.data(), c = c.data()] __device__(int i) { c[i] = a[i] + b[i]; });

    std::cout << "the sum of " << a << " and " << b << " is " << c << std::endl;
}

void sum_vector_elements() {
    cuda_vector<int> a(1024);
    std::iota(a.begin(), a.end(), 0);

    cuda_vector<int> b(1);

    cuda_parallel_for(a.size(), [a = a.data(), b = b.data()] __device__(int i) { atomicAdd(b, a[i]); });

    std::cout << "the sum of 0~" << a.size() << " is " << b[0] << std::endl;
}

int main() {
    two_vector_add();
    sum_vector_elements();
    return 0;
}