#include <iostream>
#include <vector>
#include <openacc.h>

int main() {
    const int N = 100;
    std::vector<float> data(N);

    // Initialize the array
    for (int i = 0; i < N; i++) {
        data[i] = i + 1; // 1,2,3,...,N
    }

    float total = 0.0f;

    // Parallel reduction using OpenACC
    #pragma acc parallel loop reduction(+:total) copyin(data[0:N])
    for (int i = 0; i < N; i++) {
        total += data[i];
    }

    std::cout << "Sum of array = " << total << "\n";

    return 0;
}
