#include <cstdio>
#include <cmath>

#define __noinline__ __attribute__((noinline))
#define __forceinline__ __attribute__((always_inline))
'''
#define __noinline__ custom_noinline
#define __forceinline__ custom_forceinline
'''
#define LOG(x) printf("LOG: %d\n", x)

__global__ void kernel_inline(int *arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = i * i;
    }
}

__host__ __forceinline__ int add(int a, int b) {
    return a + b;
}

__host__ __noinline__ int multiply(int a, int b) {
    return a * b;
}

int main() {
    int n = 10;
    int arr[10];
    kernel_inline<<<1, 1>>>(arr, n);
    printf("Sum: %d\n", add(3, 4));
    printf("Product: %d\n", multiply(3, 4));
    LOG(n);
    return 0;
}
