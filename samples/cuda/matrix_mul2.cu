// file: divergence_calc_float.cu
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <vector>

using FloatType = float;

// ------------------ Kernel: Compute Divergence ------------------
__global__ void computeDivergence(const FloatType* __restrict__ fieldX,
                                  const FloatType* __restrict__ fieldY,
                                  const FloatType* __restrict__ fieldZ,
                                  FloatType* __restrict__ divergence,
                                  int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = k * nx * ny + j * nx + i;

    FloatType centerX = fieldX[idx];
    FloatType centerY = fieldY[idx];
    FloatType centerZ = fieldZ[idx];

    FloatType fx = (i < nx-1 ? fieldX[idx+1] : centerX) - (i > 0 ? fieldX[idx-1] : centerX);
    FloatType fy = (j < ny-1 ? fieldY[idx+nx] : centerY) - (j > 0 ? fieldY[idx-nx] : centerY);
    FloatType fz = (k < nz-1 ? fieldZ[idx+nx*ny] : centerZ) - (k > 0 ? fieldZ[idx-nx*ny] : centerZ);

    divergence[idx] = 0.5f * (fx + fy + fz);
}

// ------------------ Host Utility ------------------
FloatType sumField(const std::vector<FloatType>& hostVec) {
    thrust::device_vector<FloatType> d_vec = hostVec;
    return thrust::reduce(d_vec.begin(), d_vec.end(), FloatType(0));
}

// ------------------ Main ------------------
int main() {
    constexpr int NX = 32, NY = 32, NZ = 32;
    int N = NX * NY * NZ;

    std::vector<FloatType> h_fieldX(N), h_fieldY(N), h_fieldZ(N);
    (int k=0;k<NZ;k++)
        for(int j=0;j<NY;j++)
            for(int i=0;i<NX;i++){
                int idx = k*NX*NY + j*NX + i;
                h_fieldX[idx] = float(i);
                h_fieldY[idx] = float(j);
                h_fieldZ[idx] = float(k);
            }

    FloatType *d_fieldX, *d_fieldY, *d_fieldZ, *d_div;
    cudaMalloc(&d_fieldX, N*sizeof(FloatType));
    cudaMalloc(&d_fieldY, N*sizeof(FloatType));
    cudaMalloc(&d_fieldZ, N*sizeof(FloatType));
    cudaMalloc(&d_div, N*sizeof(FloatType));

    cudaMemcpy(d_fieldX, h_fieldX.data(), N*sizeof(FloatType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fieldY, h_fieldY.data(), N*sizeof(FloatType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fieldZ, h_fieldZ.data(), N*sizeof(FloatType), cudaMemcpyHostToDevice);

    dim3 block(8,8,8);
    dim3 grid((NX+7)/8, (NY+7)/8, (NZ+7)/8);
    computeDivergence<<<grid, block>>>(d_fieldX, d_fieldY, d_fieldZ, d_div, NX, NY, NZ);
    cudaDeviceSynchronize();

    std::vector<FloatType> h_div(N);
    cudaMemcpy(h_div.data(), d_div, N*sizeof(FloatType), cudaMemcpyDeviceToHost);

    FloatType totalDivergence = sumField(h_div);
    std::cout << "Total divergence: " << totalDivergence << std::endl;

    cudaFree(d_fieldX);
    cudaFree(d_fieldY);
    cudaFree(d_fieldZ);
    cudaFree(d_div);
    return 0;
}
