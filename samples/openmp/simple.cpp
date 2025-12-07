#include <iostream>
#include <omp.h>

int main() {
    const int N = 10;
    int data[N];

    // Initialize array
    for(int i = 0; i < N; i++) {
        data[i] = i + 1;  // 1,2,3,...10
    }

    int total = 0;

    // Parallel reduction using OpenMP
    #pragma omp parallel for reduction(+:total)
    for(int i = 0; i < N; i++) {
        total += data[i];
    }

    std::cout << "Number of threads used: "
              << omp_get_max_threads() << "\n";

    std::cout << "Sum of array = " << total << "\n";

    return 0;
}
