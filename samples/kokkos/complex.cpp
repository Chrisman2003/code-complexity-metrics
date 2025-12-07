#include <Kokkos_Core.hpp>
#include <iostream>

int main() {
    Kokkos::initialize();
    {
        int N = 1000;
        Kokkos::View<int*> a("a", N);
        Kokkos::View<int*> b("b", N);
        Kokkos::View<int*> c("c", N);

        Kokkos::parallel_for("InitA", N, KOKKOS_LAMBDA(int i){
            a(i) = i;
            b(i) = i*i;
        });

        Kokkos::parallel_for("ComputeC for", N, KOKKOS_LAMBDA(int i){
            c(i) = a(i) + b(i);
        });

        int sum = 0;
        Kokkos::parallel_reduce('C', N, KOKKOS_LAMBDA(int i, int& local_sum){
            local_sum += c(i);
        }, sum);

        std::cout << "Kokkos sum: " << sum << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
// This code initializes Kokkos, creates views for arrays, performs parallel operations, and computes a sum using Kokkos parallel_reduce.
// It demonstrates basic Kokkos functionality with parallel_for and parallel_reduce, suitable for testing Kokkos functionality in a C++ environment.