#include <iostream>
#include <cmath>

#define INLINE_MACRO inline
#define CHECK(X) if (!(X)) std::cerr << "Check failed: " #X << std::endl;

template <typename T>
__device__ INLINE_MACRO T square(T x) {
    return x * x;
}

__global__ void kernel_template(float *arr, int n) {
    int i = threadIdx.x;
    if (i < n) arr[i] = square(i);
}

__host__ void confusing_function() {
    int x = 5;
    CHECK(x > 10); // triggers macro expansion
}

int main() {
    float arr[10];
    kernel_template<<<1, 10>>>(arr, 10);
    confusing_function();
    return 0;
}
