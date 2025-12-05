#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>

int main() {
    const int N = 10;

    // 1) Host vectors
    thrust::host_vector<float> h_A(N), h_B(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    // 2) Transfer to device vectors
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(N);

    // 3) Elementwise addition on GPU
    thrust::transform(d_A.begin(), d_A.end(),
                      d_B.begin(),
                      d_C.begin(),
                      thrust::plus<float>());

    // 4) Copy result back to host
    thrust::host_vector<float> h_C = d_C;

    // 5) Print results
    std::cout << "C = A + B:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_C[i] << " ";
    std::cout << "\n";

    return 0;
}
