#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "nonexistent_header.h" // simulate missing header
#include <stdio.h>

__global__ void compute_squares(int *data, int n) {
    int idx = threadIdx.x;
    if (idx < n) data[idx] = idx * idx;
}

__device__ int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main() {
    int data[5];
    compute_squares<<<1, 5>>>(data, 5);

    int f = factorial(5); // device function call from host? illegal but triggers analyzer
    printf("Factorial: %d\n", f);
    return 0;
}
