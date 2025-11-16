#include <iostream>
using namespace std;
void partition(int A[], int F, int L, int& pivotIndex);

void partition(int A[], int F, int L, int& pivotIndex)
{
    int pivot = A[F];
    int lastS1 = F;
    int firstUnknown = F + 1;

    for (; firstUnknown <= L; ++firstUnknown) {
        if (A[firstUnknown] < pivot) {
            ++lastS1;
            Swap(A[firstUnknown], A[lastS1]);
        }
    }

    Swap(A[F], A[lastS1]);
    pivotIndex = lastS1;
}