// file: advanced_cuda.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>

#define N 1024 * 1024       // 1M elements
#define THREADS 256
#define BLOCKS ((N + THREADS - 1)/THREADS)

// Macro for CUDA error checking
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Kernel: element-wise addition of two __half arrays
__global__ void vectorAddHalf(const __half* A, const __half* B, __half* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = __hadd(A[idx], B[idx]);
    }
}

// Kernel: reduction (sum) using shared memory
__global__ void reduceSum(const __half* input, float* output, int n) {
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.0f;
    if (idx < n) temp = __half2float(input[idx]);
    sdata[tid] = temp;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block result to output
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    __half *h_A, *h_B, *h_C;
    __half *d_A, *d_B, *d_C;
    float *d_partialSum, *h_partialSum;

    size_t size = N * sizeof(__half);
    size_t partialSize = BLOCKS * sizeof(float);

    // Allocate host memory
    h_A = new __half[N];
    h_B = new __half[N];
    h_C = new __half[N];
    h_partialSum = new float[BLOCKS];

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = __float2half(static_cast<float>(i % 100) / 100.0f);
        h_B[i] = __float2half(static_cast<float>((i*2) % 100) / 100.0f);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    CUDA_CHECK(cudaMalloc((void**)&d_partialSum, partialSize));

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    auto start = std::chrono::high_resolution_clock::now();

    // Async copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Launch vectorAddHalf kernel
    vectorAddHalf<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reduction kernel to sum the result
    reduceSum<<<BLOCKS, THREADS>>>(d_C, d_partialSum, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_partialSum, d_partialSum, partialSize, cudaMemcpyDeviceToHost));

    // Compute final sum on host
    float totalSum = 0.0f;
    for (int i = 0; i < BLOCKS; i++) totalSum += h_partialSum[i];

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Total sum of vector (half precision): " << totalSum << std::endl;
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_partialSum;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_partialSum));

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}
