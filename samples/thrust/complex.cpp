#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <iostream>
#include <cmath>

// Custom functor to compute weighted sum
struct weighted_sum
{
    __host__ __device__
    float operator()(const thrust::tuple<float, float>& t) const {
        float x = thrust::get<0>(t);
        float w = thrust::get<1>(t);
        return x * w;
    }
};

// Custom unary functor for squaring a number
struct square_functor
{
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

int main() {
    const size_t N = 1024;

    // -------------------------------
    // 1) Host vectors
    // -------------------------------
    thrust::host_vector<float> h_X(N);
    thrust::host_vector<float> h_W(N);

    for (size_t i = 0; i < N; ++i) {
        h_X[i] = float(i) * 0.5f;          // example data
        h_W[i] = 1.0f + 0.01f * i;         // weights
    }

    // -------------------------------
    // 2) Transfer to device
    // -------------------------------
    thrust::device_vector<float> d_X = h_X;
    thrust::device_vector<float> d_W = h_W;
    thrust::device_vector<float> d_Y(N);
    thrust::device_vector<float> d_Z(N);

    // -------------------------------
    // 3) Elementwise operations
    //    Y = X^2
    // -------------------------------
    thrust::transform(d_X.begin(), d_X.end(), d_Y.begin(), square_functor());

    // -------------------------------
    // 4) Weighted sum: Z = X * W
    // -------------------------------
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_X.begin(), d_W.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_X.end(),   d_W.end())),
        d_Z.begin(),
        weighted_sum()
    );

    // -------------------------------
    // 5) Reduce operations
    //    sum_Y = sum(Y), sum_Z = sum(Z)
    // -------------------------------
    float sum_Y = thrust::reduce(d_Y.begin(), d_Y.end(), 0.0f, thrust::plus<float>());
    float sum_Z = thrust::reduce(d_Z.begin(), d_Z.end(), 0.0f, thrust::plus<float>());

    std::cout << "Sum of squares (Y) = " << sum_Y << "\n";
    std::cout << "Weighted sum (Z)   = " << sum_Z << "\n";

    // -------------------------------
    // 6) Sort X descending
    // -------------------------------
    thrust::sort(d_X.begin(), d_X.end(), thrust::greater<float>());

    // -------------------------------
    // 7) Remove duplicates
    // -------------------------------
    auto new_end = thrust::unique(d_X.begin(), d_X.end());
    d_X.erase(new_end, d_X.end());

    // -------------------------------
    // 8) Count elements > 100
    // -------------------------------
    int count_gt_100 = thrust::count_if(d_X.begin(), d_X.end(), [] __host__ __device__ (float x){ return x > 100.0f; });
    std::cout << "Number of elements > 100 after sort and unique: " << count_gt_100 << "\n";

    // -------------------------------
    // 9) Print first 5 elements of X
    // -------------------------------
    thrust::host_vector<float> h_X_result = d_X;
    std::cout << "First 5 elements of X (sorted & unique): ";
    for (size_t i = 0; i < std::min(size_t(5), h_X_result.size()); ++i)
        std::cout << h_X_result[i] << " ";
    std::cout << "\n";

    return 0;
}
