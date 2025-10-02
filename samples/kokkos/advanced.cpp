#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>

struct ComplexComputation {
    Kokkos::View<double*> data;
    Kokkos::View<double*[3]> vectors;

    ComplexComputation(int N) : data("data", N), vectors("vectors", N) {
        Kokkos::parallel_for("InitVectors", N, KOKKOS_LAMBDA(int i){
            vectors(i,0) = i * 0.1;
            vectors(i,1) = i * 0.2;
            vectors(i,2) = i * 0.3;
        });
    }

    void run(int N) {
        // Compute magnitudes into data
        Kokkos::parallel_for("ComputeMagnitudes", N, KOKKOS_LAMBDA(int i){
            double x = vectors(i,0);
            double y = vectors(i,1);
            double z = vectors(i,2);
            data(i) = std::sqrt(x*x + y*y + z*z);
        });

        // Perform a complex reduction
        double total_sum = 0.0;
        Kokkos::parallel_reduce("ComplexSum", N, KOKKOS_LAMBDA(int i, double& local_sum){
            double val = data(i);
            for(int j=0; j<3; ++j){
                val += std::sin(vectors(i,j)) * std::cos(vectors(i,j));
            }
            local_sum += val;
        }, total_sum);

        std::cout << "Total sum: " << total_sum << std::endl;

        // Nested parallel_for example
        Kokkos::parallel_for("NestedLoops", N, KOKKOS_LAMBDA(int i){
            for(int j=0; j<3; ++j){
                data(i) += std::exp(-vectors(i,j));
            }
        });
    }
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N = 1000000;  // Number of elements
        ComplexComputation comp(N);
        comp.run(N);
    }
    Kokkos::finalize();
    return 0;
}
