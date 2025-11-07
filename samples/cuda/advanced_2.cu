#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while(0)

using FloatType = float;
using Vector3 = float3;

// ------------------ Device Utility Functions ------------------
__device__ Vector3 operator+(Vector3 a, Vector3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ Vector3 operator-(Vector3 a, Vector3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ Vector3 operator*(Vector3 a, FloatType b) {
    return {a.x * b, a.y * b, a.z * b};
}

__device__ FloatType dot(Vector3 a, Vector3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// ------------------ Kernel: Compute Divergence ------------------
__global__ void computeDivergence(
        const Vector3* __restrict__ field,
        FloatType* __restrict__ divergence,
        int nx, int ny, int nz) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= nx || j >= ny || k >= nz) return;

    int idx = k*nx*ny + j*nx + i;

    Vector3 center = field[idx];
    Vector3 fx = (i < nx-1 ? field[idx+1] : center) - (i > 0 ? field[idx-1] : center);
    Vector3 fy = (j < ny-1 ? field[idx+nx] : center) - (j > 0 ? field[idx-nx] : center);
    Vector3 fz = (k < nz-1 ? field[idx+nx*ny] : center) - (k > 0 ? field[idx-nx*ny] : center);

    divergence[idx] = 0.5f * (fx.x + fy.y + fz.z); // finite difference approximation
}

// ------------------ Host Template Function ------------------
template<typename T>
T sumField(const std::vector<T>& hostVec) {
    thrust::device_vector<T> d_vec = hostVec;
    return thrust::reduce(d_vec.begin(), d_vec.end(), T(0));
}

// ------------------ Main ------------------
int main() {
    constexpr int NX = 32, NY = 32, NZ = 32;
    int N = NX*NY*NZ;

    std::vector<Vector3> h_field(N);
    for(int k=0;k<NZ;k++)
        for(int j=0;j<NY;j++)
            for(int i=0;i<NX;i++){
                int idx = k*NX*NY + j*NX + i;
                h_field[idx] = {float(i), float(j), float(k)};
            }

    Vector3* d_field;
    FloatType* d_div;
    CHECK_CUDA(cudaMalloc(&d_field, N*sizeof(Vector3)));
    CHECK_CUDA(cudaMalloc(&d_div, N*sizeof(FloatType)));
    CHECK_CUDA(cudaMemcpy(d_field, h_field.data(), N*sizeof(Vector3), cudaMemcpyHostToDevice));

    dim3 block(8,8,8);
    dim3 grid((NX+7)/8, (NY+7)/8, (NZ+7)/8);
    computeDivergence<<<grid, block>>>(d_field, d_div, NX, NY, NZ);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<FloatType> h_div(N);
    CHECK_CUDA(cudaMemcpy(h_div.data(), d_div, N*sizeof(FloatType), cudaMemcpyDeviceToHost));

    FloatType totalDivergence = sumField(h_div);
    std::cout << "Total divergence: " << totalDivergence << std::endl;

    cudaFree(d_field);
    cudaFree(d_div);
    return 0;
}
