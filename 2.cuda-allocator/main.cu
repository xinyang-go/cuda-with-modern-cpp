#include <iostream>
#include <vector>

#include "cuda_allocator.hpp"

template <typename T>
using cuda_vector = std::vector<T, cuda_alloctor<T>>;

template <typename T, typename A>
std::ostream &operator<<(std::ostream &os, const std::vector<T, A> &vec) {
    if (vec.empty()) return os << "[]";
    os << "[" << vec[0];
    for (size_t i = 1; i < vec.size(); i++) os << ", " << vec[i];
    return os << "]";
}

int main() {
    cuda_vector<int> x{0, 1, 2, 3};
    
    std::cout << "create cuda_vector = " << x << std::endl;

    return 0;
}