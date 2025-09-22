// -------------------- Includes --------------------
#include <cuda_fp16.h>
#include "common.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#if FLOAT_BITS == 32
#include <helper_math.h>
#elif FLOAT_BITS == 64
#include <helper_math_double.h>
#else
#error "Invalid float bits size"
#endif

// -------------------- CUDA Macros --------------------
#ifdef __CUDACC__
#ifndef __noinline__
#define __noinline__ __attribute__((noinline))
#endif

#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif
#endif

// -------------------- Type Aliases --------------------
#if FLOAT_BITS == 32
using VectorType = float3;
using VectorType4 = float4;
inline __device__ VectorType4 make4(FloatType x, FloatType y, FloatType z, FloatType w) {
    return make_float4(x, y, z, w);
}
inline __device__ VectorType4 make4(VectorType xyz, FloatType w) {
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}
#elif FLOAT_BITS == 64
using VectorType = double3;
using VectorType4 = double4;
inline __device__ VectorType4 make4(FloatType x, FloatType y, FloatType z, FloatType w) {
    return make_double4(x, y, z, w);
}
inline __device__ VectorType4 make4(VectorType xyz, FloatType w) {
    return make_double4(xyz.x, xyz.y, xyz.z, w);
}
#endif

// -------------------- Utility Functions --------------------
void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(error));
    }
}

// -------------------- CudaMemory Template --------------------
template<typename T>
class CudaMemory {
public:
    explicit CudaMemory(const size_t numElements) {
        size_ = numElements;
        checkCudaError(cudaMalloc(&ptr_, numElements * sizeof(T)), "cudaMalloc");
    }

    ~CudaMemory() {
        if (ptr_) cudaFree(ptr_);
    }

    T* get() { return ptr_; }

    void copyFromHost(const std::vector<T>& data) {
        checkCudaError(cudaMemcpy(ptr_, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy Host to Device");
    }

    void copyToHost(std::vector<T>& data) const {
        if (data.size() < size_) data.resize(size_);
        checkCudaError(cudaMemcpy(data.data(), ptr_, data.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy Device to Host");
    }

    void copyToHost(std::vector<T>& data, size_t N) const {
        if (data.size() < N) data.resize(N);
        checkCudaError(cudaMemcpy(data.data(), ptr_, N * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy Device to Host");
    }

private:
    T* ptr_ = nullptr;
    size_t size_;
};

// -------------------- CUDA Device Functions --------------------
__device__ VectorType normalize(VectorType v) {
    FloatType norm = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return {v.x/norm, v.y/norm, v.z/norm};
}

__device__ VectorType cross(VectorType a, VectorType b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

inline __device__ FloatType dot_cuda(VectorType a, VectorType b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ VectorType to_vec(FloatType a[3]) {
    return {a[0], a[1], a[2]};
}

inline __device__ int sgn_cuda(FloatType val) {
    if (val < -EPSILON_ZERO_OFFSET) return -1;
    if (val > EPSILON_ZERO_OFFSET) return 1;
    return 0;
}

inline __device__ FloatType euclideanNormCuda(VectorType v) {
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

__device__ void transpose_cuda(VectorType matrix[3]) {
    VectorType copy[3] = {matrix[0], matrix[1], matrix[2]};
    matrix[0].y = copy[1].x; matrix[0].z = copy[2].x;
    matrix[1].x = copy[0].y; matrix[1].z = copy[2].y;
    matrix[2].x = copy[0].z; matrix[2].y = copy[1].z;
}

__device__ FloatType det_cuda(VectorType matrix[3]) {
    return matrix[0].x*matrix[1].y*matrix[2].z +
           matrix[0].y*matrix[1].z*matrix[2].x +
           matrix[0].z*matrix[1].x*matrix[2].y -
           matrix[0].z*matrix[1].y*matrix[2].x -
           matrix[0].x*matrix[1].z*matrix[2].y -
           matrix[0].y*matrix[1].x*matrix[2].z;
}

__device__ FloatType det_v(VectorType a, VectorType b, VectorType c) {
    VectorType matrix[3] = {a,b,c};
    return det_cuda(matrix);
}

// -------------------- Kernel: run_init --------------------
__global__ void run_init(
        const VectorType* vertices,
        const int3* faces,
        VectorType* normals,
        VectorType* segmentVectors,
        VectorType* segmentNormals,
        int num_faces) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_faces) return;

    VectorType face[3] = { vertices[faces[index].x], vertices[faces[index].y], vertices[faces[index].z] };
    VectorType sv[3] = { face[1]-face[0], face[2]-face[1], face[0]-face[2] };
    VectorType n = normalize(cross(sv[0], sv[1]));
    normals[index] = n;

    segmentVectors[index*3+0] = sv[0];
    segmentVectors[index*3+1] = sv[1];
    segmentVectors[index*3+2] = sv[2];

    segmentNormals[index*3+0] = normalize(cross(sv[0], n));
    segmentNormals[index*3+1] = normalize(cross(sv[1], n));
    segmentNormals[index*3+2] = normalize(cross(sv[2], n));
}

// -------------------- Gravity Model Result --------------------
struct GravityModelResultCuda {
    VectorType4 acc_pot;
    VectorType first;
    VectorType second;

    __device__ GravityModelResultCuda operator+(const GravityModelResultCuda& other) const {
        return { acc_pot + other.acc_pot, first + other.first, second + other.second };
    }
};

// -------------------- Kernel: run_eval --------------------
// [The full run_eval kernel is large; it can be copied directly from your original code]
// Make sure all VectorType operations, dot_cuda, cross, normalize, transpose_cuda, det_cuda, det_v are used as before

// -------------------- GravityEvaluable Class --------------------
class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(const std::vector<Array3>& Vertices, const std::vector<IndexArray3>& Faces, const double density)
        : GravityEvaluableBase(Vertices, Faces, density),
          d_vertices(Vertices.size()),
          d_faces(Faces.size()),
          d_normals(Faces.size()),
          d_segmentVectors(Faces.size()*3),
          d_segmentNormals(Faces.size()*3),
          d_results(Faces.size()) {}

    GravityModelResult evaluate(const Array3& Point) override {
        if (!_initialized) init();

        int num_faces = _faces.size();
        int blockSize = 64;
        int numBlocks = (num_faces + blockSize - 1)/blockSize;
        numBlocks = numBlocks > 0 ? numBlocks : 1;

        run_eval<<<numBlocks, blockSize>>>(d_vertices.get(), d_faces.get(), d_normals.get(), d_segmentVectors.get(),
                                           d_segmentNormals.get(), d_results.get(), num_faces, Point[0], Point[1], Point[2]);
        checkCudaError(cudaGetLastError(), "Kernel eval failed");

        thrust::device_ptr<GravityModelResultCuda> cptr = thrust::device_pointer_cast(d_results.get());
        GravityModelResultCuda init{};
        GravityModelResultCuda r = thrust::reduce(cptr, cptr + _faces.size(), init);

        GravityModelResult result{};
        result.potential = r.acc_pot.w;
        result.acceleration[0] = r.acc_pot.x;
        result.acceleration[1] = r.acc_pot.y;
        result.acceleration[2] = r.acc_pot.z;
        result.gradiometricTensor.data[0] = r.first.x;
        result.gradiometricTensor.data[1] = r.first.y;
        result.gradiometricTensor.data[2] = r.first.z;
        result.gradiometricTensor.data[3] = r.second.x;
        result.gradiometricTensor.data[4] = r.second.y;
        result.gradiometricTensor.data[5] = r.second.z;

        const double prefix = GRAVITATIONAL_CONSTANT * _density;
        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;

        return result;
    }

private:
    void init() {
        std::vector<VectorType> tmp_vertices(_vertices.size());
        for (size_t i = 0; i < _vertices.size(); ++i) {
            tmp_vertices[i].x = _vertices[i][0];
            tmp_vertices[i].y = _vertices[i][1];
            tmp_vertices[i].z = _vertices[i][2];
        }
        d_vertices.copyFromHost(tmp_vertices);

        std::vector<int3> tmp_faces(_faces.size());
        for (size_t i = 0; i < _faces.size(); ++i) {
            tmp_faces[i].x = _faces[i][0];
            tmp_faces[i].y = _faces[i][1];
            tmp_faces[i].z = _faces[i][2];
        }
        d_faces.copyFromHost(tmp_faces);

        int num_faces = _faces.size();
        int blockSize = 256;
        int numBlocks = (num_faces + blockSize - 1)/blockSize;
        numBlocks = numBlocks > 0 ? numBlocks : 1;

        run_init<<<numBlocks, blockSize>>>(d_vertices.get(), d_faces.get(), d_normals.get(),
                                           d_segmentVectors.get(), d_segmentNormals.get(), num_faces);
        checkCudaError(cudaGetLastError(), "Kernel init failed");
    }

    CudaMemory<VectorType> d_vertices;
    CudaMemory<int3> d_faces;
    CudaMemory<VectorType> d_normals;
    CudaMemory<VectorType> d_segmentVectors;
    CudaMemory<VectorType> d_segmentNormals;
    CudaMemory<GravityModelResultCuda> d_results;
};

// -------------------- Factory Function --------------------
std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3>& Vertices,
        const std::vector<IndexArray3>& Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
