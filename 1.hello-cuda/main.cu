#include <cuda.h>
#include <cstdio>
#include <iostream>

__global__ void cuda_hello() {
    printf("hello cuda!\n");

    /* compile error if uncomment the following line */ 
    // std::cout << "hello cuda!" << std::endl;
}

int main() {
    cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}