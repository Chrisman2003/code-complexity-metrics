#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <openacc.h>

constexpr int N = 512;        // Matrix size
constexpr int ARR_SIZE = 200000;   // Array size for merge sort & prefix sum

// Helper to access flat arrays as 2D
inline float& A(std::vector<float>& M, int r, int c, int n = N) {
    return M[r * n + c];
}

// Timing utility
double now() { return std::chrono::high_resolution_clock::now().time_since_epoch().count() * 1e-9; }

// ============================================================================
// Parallel prefix sum using OpenACC
// ============================================================================
void parallel_prefix_sum(std::vector<int>& arr) {
    int n = arr.size();
    std::vector<int> partial_sums(ARR_SIZE, 0);

    // Step 1: compute block-wise sums
    #pragma acc parallel loop
    for (int i = 0; i < n; i++) {
        if (i == 0) arr[i] = arr[i];
        else arr[i] += arr[i-1];
    }
}

// ============================================================================
// Merge utility for merge sort
// ============================================================================
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for(int i = 0; i < n1; i++) L[i] = arr[left + i];
    for(int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i=0, j=0, k=left;
    while(i<n1 && j<n2) arr[k++] = (L[i]<=R[j])?L[i++]:R[j++];
    while(i<n1) arr[k++] = L[i++];
    while(j<n2) arr[k++] = R[j++];
}

// ============================================================================
// Recursive merge sort with OpenACC tasks
// ============================================================================
void parallel_merge_sort(std::vector<int>& arr, int left, int right, int depth=0) {
    if(left>=right) return;

    int mid = left + (right-left)/2;

    if(depth < 4) { // limit task creation depth
        #pragma acc parallel loop
        for(int i=left;i<=mid;i++) arr[i] = arr[i];
        #pragma acc parallel loop
        for(int i=mid+1;i<=right;i++) arr[i] = arr[i];

        parallel_merge_sort(arr, left, mid, depth+1);
        parallel_merge_sort(arr, mid+1, right, depth+1);
    } else {
        parallel_merge_sort(arr, left, mid, depth+1);
        parallel_merge_sort(arr, mid+1, right, depth+1);
    }

    merge(arr, left, mid, right);
}

// ============================================================================
// Matrix multiplication using blocking
// ============================================================================
void matmul(const std::vector<float>& A,
            const std::vector<float>& B,
            std::vector<float>& C,
            int n) 
{
    const int BLOCK = 32;

    #pragma acc data copyin(A[0:n*n],B[0:n*n]) copy(C[0:n*n])
    {
        #pragma acc parallel loop collapse(2) gang vector
        for(int ii = 0; ii < n; ii += BLOCK) {
            for(int jj = 0; jj < n; jj += BLOCK) {
                for(int i = ii; i < std::min(ii+BLOCK,n); i++) {
                    for(int j = jj; j < std::min(jj+BLOCK,n); j++) {
                        float sum = 0.0f;
                        for(int k=0;k<n;k++) {
                            sum += A[i*n+k]*B[k*n+j];
                        }
                        C[i*n+j] = sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Main program
// ============================================================================
int main() {

    std::cout << "=== COMPLEX OPENACC PROGRAM ===\n";

    // -------------------------------
    // PART 1: Merge Sort
    // -------------------------------
    std::vector<int> arr(ARR_SIZE);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0,1000000);
    for(int &x : arr) x = dist(rng);

    std::cout << "Sorting " << ARR_SIZE << " integers...\n";
    auto t1 = std::chrono::high_resolution_clock::now();

    #pragma acc parallel
    {
        #pragma acc single
        parallel_merge_sort(arr,0,arr.size()-1);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Sort completed in " 
              << std::chrono::duration<double>(t2-t1).count() << " seconds\n";

    if(!std::is_sorted(arr.begin(),arr.end())) std::cout << "❌ Sort failed!\n";
    else std::cout << "✔ Sort validated.\n";

    // -------------------------------
    // PART 2: Prefix Sum
    // -------------------------------
    std::cout << "Running parallel prefix sum...\n";
    auto t3 = std::chrono::high_resolution_clock::now();

    parallel_prefix_sum(arr);

    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Prefix sum completed in " 
              << std::chrono::duration<double>(t4-t3).count() << " seconds\n";

    // -------------------------------
    // PART 3: Matrix Multiplication
    // -------------------------------
    std::vector<float> A_mat(N*N), B_mat(N*N), C_mat(N*N,0.0f);

    for(int i=0;i<N*N;i++) {
        A_mat[i] = (i%15)*0.5f;
        B_mat[i] = (i%9)*0.25f;
    }

    std::cout << "Multiplying " << N << "x" << N << " matrices...\n";
    auto t5 = std::chrono::high_resolution_clock::now();

    matmul(A_mat,B_mat,C_mat,N);

    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "Matmul completed in " 
              << std::chrono::duration<double>(t6-t5).count() << " seconds\n";

    // Quick validation: sum of all elements
    float checksum = 0.0f;
    #pragma acc parallel loop reduction(+:checksum)
    for(int i=0;i<N*N;i++)
        checksum += C_mat[i];

    std::cout << "Checksum = " << checksum << "\n";

    std::cout << "=== PROGRAM DONE ===\n";

    return 0;
}
