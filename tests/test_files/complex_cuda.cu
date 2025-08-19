#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000;
    float *h_a, *h_b, *h_c;
    cudaMallocHost((void**)&h_a, n * sizeof(float));
    cudaMallocHost((void**)&h_b, n * sizeof(float));
    cudaMallocHost((void**)&h_c, n * sizeof(float));

    for(int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = i*i; }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n*sizeof(float));
    cudaMalloc((void**)&d_b, n*sizeof(float));
    cudaMalloc((void**)&d_c, n*sizeof(float));

    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(float), cudaMemcpyHostToDevice);

    vectorAdd<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);

    printf("Finished CUDA Vector Addition\n");
    return 0;
}
// This code is a simple CUDA kernel for vector addition.
