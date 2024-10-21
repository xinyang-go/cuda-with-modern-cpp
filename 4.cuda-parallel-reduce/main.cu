#include <cassert>
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

template <typename T, typename F>
void __global__ cuda_parallel_reduce_kernel(const T *data, int n, T *output, T init, F op) {
    assert(blockDim.x <= 512);

    int b = blockIdx.x, t = threadIdx.x;
    int idx = b * blockDim.x + t;
    int step = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += step) {
        init = op(init, data[i]);
    }

    extern __shared__ volatile T local[];
    local[t] = init;
    __syncthreads();
    if (t >= 256) return;
    local[t] = op(local[t], local[t + 256]);
    __syncthreads();
    if (t >= 128) return;
    local[t] = op(local[t], local[t + 128]);
    __syncthreads();
    if (t >= 64) return;
    local[t] = op(local[t], local[t + 64]);
    __syncthreads();
    if (t >= 32) return;
    local[t] = op(local[t], local[t + 32]);
    local[t] = op(local[t], local[t + 16]);
    local[t] = op(local[t], local[t + 8]);
    local[t] = op(local[t], local[t + 4]);
    local[t] = op(local[t], local[t + 2]);
    local[t] = op(local[t], local[t + 1]);
    output[b] = local[0];
}

template <typename T, typename F>
void cuda_parallel_reduce(const cuda_vector<T> &vec, cuda_vector<T> &output, int init, F &&op,
                                  int gridDim = 128, int blockDim = 512) {
    output.resize(gridDim * blockDim, init);
    CheckCudaKernel(cuda_parallel_reduce_kernel<<<gridDim, blockDim, blockDim * sizeof(T)>>>(
        vec.data(), vec.size(), output.data(), init, op));
    CheckCudaApi(cudaDeviceSynchronize());
    for (size_t i = 1; i < output.size(); i++) output[0] = op(output[0], output[i]);
    output.resize(1);
}



int main() {
    cuda_vector<int> x(1 << 24);  // 16MB
    std::iota(x.begin(), x.end(), 1);

    cuda_vector<int> sum, max, min;
    cuda_parallel_reduce(x, sum, 0, [] __device__ __host__(int a, int b) { return a + b; });
    cuda_parallel_reduce(x, max, std::numeric_limits<int>::min(),
                                 [] __device__ __host__(int a, int b) { return std::max(a, b); });
    cuda_parallel_reduce(x, min, std::numeric_limits<int>::max(),
                                 [] __device__ __host__(int a, int b) { return std::min(a, b); });
    std::cout << "reduce sum of 1~" << x.size() << " is " << sum[0] << std::endl;
    std::cout << "reduce max of 1~" << x.size() << " is " << max[0] << std::endl;
    std::cout << "reduce min of 1~" << x.size() << " is " << min[0] << std::endl;

    return 0;
}