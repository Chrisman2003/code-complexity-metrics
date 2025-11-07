// file: matrix_mul.cu
#include <cuda_runtime.h>
#include <iostream>

#define N 1024       // Matrix size (NxN)
#define TILE_WIDTH 16 // Tile size for shared memory

// CUDA kernel: tiled matrix multiplication
__global__ void matrixMulShared(float* A, float* B, float* C, int n) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0;

    // Loop over tiles
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load tiles into shared memory
        if (row < n && t * TILE_WIDTH + threadIdx.x < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && t * TILE_WIDTH + threadIdx.y < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply tile elements
        for (int i = 0; i < TILE_WIDTH; i++)
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = value;
}

// Utility function to initialize matrices
void initMatrix(float* mat, int n) {
    for (int i = 0; i < n * n; i++)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

// Main function
int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];

    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print first 5x5 block
    std::cout << "C[0..4][0..4]:" << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            std::cout << h_C[i * N + j] << " ";
        std::cout << std::endl;
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
