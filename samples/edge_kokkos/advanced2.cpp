#include <Kokkos_Core.hpp>
#include <iostream>

// Custom type for reduction
struct ComplexResult {
    double sum;
    double square_sum;

    KOKKOS_INLINE_FUNCTION
    ComplexResult() : sum(0.0), square_sum(0.0) {}

    KOKKOS_INLINE_FUNCTION
    ComplexResult& operator+=(const ComplexResult& other) {
        sum += other.sum;
        square_sum += other.square_sum;
        return *this;
    }
};

// Define a reducer for ComplexResult
struct ComplexResultReducer {
    typedef ComplexResult value_type;

    KOKKOS_INLINE_FUNCTION
    static void join(volatile value_type& dest, const volatile value_type& src) {
        dest += src;
    }

    KOKKOS_INLINE_FUNCTION
    static void init(value_type& val) {
        val.sum = 0.0;
        val.square_sum = 0.0;
    }
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int N = 1000;
        Kokkos::View<double*> data("data", N);

        // Fill data with values in parallel
        Kokkos::parallel_for("FillData", N, KOKKOS_LAMBDA(const int i){
            data(i) = i * 0.5;
        });

        // Parallel reduction using custom ComplexResult
        ComplexResult result;
        Kokkos::parallel_reduce(
            "ComputeComplex", 
            N, 
            KOKKOS_LAMBDA(const int i, ComplexResult& local){
                local.sum += data(i);
                local.square_sum += data(i) * data(i);
            }, 
            ComplexResultReducer(), 
            result
        );

        std::cout << "Sum: " << result.sum 
                  << ", Square Sum: " << result.square_sum << std::endl;

        // TeamPolicy example
        const int team_size = 32;
        const int league_size = 16;
        Kokkos::parallel_for(
            "TeamLoop", 
            Kokkos::TeamPolicy<>(league_size, team_size),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team){
                double team_sum = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, 0, N/league_size),
                    [=](const int i, double& local_sum){
                        int idx = i + team.league_rank() * (N/league_size);
                        local_sum += data(idx);
                    },
                    team_sum
                );
                if(team.team_rank() == 0){
                    printf("Team %d partial sum = %f\n", team.league_rank(), team_sum);
                }
            }
        );

    }
    Kokkos::finalize();
    return 0;
}
