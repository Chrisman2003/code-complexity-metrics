#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <omp.h>

constexpr int N = 800;        // Matrix size
constexpr int ARR_SIZE = 200000;   // Array size for merge sort and scan

// ============================================================================
// Utility function for timing
// ============================================================================
double now() { return omp_get_wtime(); }

// ============================================================================
// Prefix Sum (Scan) using OpenMP
// ============================================================================
void parallel_prefix_sum(std::vector<int>& arr) {
    int n = arr.size();
    std::vector<int> partial_sums(omp_get_max_threads(), 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int sum = 0;

        #pragma omp for schedule(static)
        for(int i = 0; i < n; i++) {
            sum += arr[i];
            arr[i] = sum;
        }

        partial_sums[tid] = sum;

        #pragma omp barrier

        int offset = 0;
        for(int i = 0; i < tid; i++)
            offset += partial_sums[i];

        #pragma omp for schedule(static)
        for(int i = 0; i < n; i++)
            arr[i] += offset;
    }
}

// ============================================================================
// Parallel Merge for Merge Sort
// ============================================================================
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for(int i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for(int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i=0, j=0, k=left;

    while(i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while(i < n1) arr[k++] = L[i++];
    while(j < n2) arr[k++] = R[j++];
}

// ============================================================================
// Recursive Merge Sort with OpenMP Tasks
// ============================================================================
void parallel_merge_sort(std::vector<int>& arr, int left, int right, int depth=0) {
    if(left >= right) return;

    int mid = left + (right - left) / 2;

    if(depth < 4) { // limit task creation depth
        #pragma omp task shared(arr)
        parallel_merge_sort(arr, left, mid, depth + 1);

        #pragma omp task shared(arr)
        parallel_merge_sort(arr, mid + 1, right, depth + 1);

        #pragma omp taskwait
    }
    else {
        parallel_merge_sort(arr, left, mid, depth + 1);
        parallel_merge_sort(arr, mid + 1, right, depth + 1);
    }

    merge(arr, left, mid, right);
}

// ============================================================================
// Matrix Multiply (blocked + OpenMP)
// ============================================================================
void matmul(const std::vector<float>& A,
            const std::vector<float>& B,
            std::vector<float>& C,
            int n) 
{
    const int BLOCK = 32;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int ii = 0; ii < n; ii += BLOCK) {
        for(int jj = 0; jj < n; jj += BLOCK) {

            for(int kk = 0; kk < n; kk += BLOCK) {
                for(int i = ii; i < std::min(ii+BLOCK, n); i++) {
                    for(int k = kk; k < std::min(kk+BLOCK, n); k++) {

                        float aik = A[i*n+k];

                        #pragma omp simd
                        for(int j = jj; j < std::min(jj+BLOCK, n); j++) {
                            C[i*n+j] += aik * B[k*n+j];
                        }
                    }
                }
            }

        }
    }
}

// ============================================================================
// Main Program
// ============================================================================
int main() {

    std::cout << "=== COMPLEX OPENMP PROGRAM ===\n";
    std::cout << "Threads: " << omp_get_max_threads() << "\n\n";

    // ============================================================================
    // PART 1: Parallel Merge Sort
    // ============================================================================
    std::vector<int> arr(ARR_SIZE);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, 1000000);

    for(int &x : arr) x = dist(rng);

    std::cout << "Sorting " << ARR_SIZE << " integers...\n";

    double t1 = now();

    #pragma omp parallel
    {
        #pragma omp single
        parallel_merge_sort(arr, 0, arr.size()-1);
    }

    double t2 = now();

    std::cout << "Sort completed in " << (t2 - t1) << " seconds.\n";

    // Validate sort
    if(!std::is_sorted(arr.begin(), arr.end()))
        std::cout << "❌ ERROR: Sort failed!\n";
    else
        std::cout << "✔ Sort validated.\n\n";

    // ============================================================================
    // PART 2: Parallel Prefix Sum
    // ============================================================================
    std::cout << "Running parallel prefix sum...\n";

    double t3 = now();
    parallel_prefix_sum(arr);
    double t4 = now();

    std::cout << "Prefix sum completed in " << (t4 - t3) << " seconds.\n\n";

    // ============================================================================
    // PART 3: Parallel Matrix Multiplication
    // ============================================================================
    std::vector<float> A(N*N), B(N*N), C(N*N, 0.0f);

    for(int i = 0; i < N*N; i++) {
        A[i] = (i % 15) * 0.5f;
        B[i] = (i % 9) * 0.25f;
    }

    std::cout << "Multiplying " << N << "x" << N << " matrices...\n";

    double t5 = now();
    matmul(A, B, C, N);
    double t6 = now();

    std::cout << "Matmul completed in " << (t6 - t5) << " seconds.\n\n";

    // Quick correctness check
    float checksum = 0.0f;
    #pragma omp parallel for reduction(+:checksum)
    for(int i = 0; i < N*N; i++)
        checksum += C[i];

    std::cout << "Checksum = " << checksum << " (for quick validation)\n";

    std::cout << "\n=== PROGRAM DONE ===\n";

    return 0;
}
