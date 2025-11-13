#include <iostream>
using namespace std;

void quicksort(int A[], int F, int L);
void partition(int A[], int F, int L, int& pivotIndex);
void Swap(int& x, int& y);

int main()
{
    int A[] = {1, 9, 0, 5, 6, 7, 8, 2, 4, 3};
    int length = 10;

    quicksort(A, 0, length - 1);

    // Print sorted result
    for (int i = 0; i < length; i++) {
        cout << A[i] << " ";
    }
    cout << endl;

    return 0;
}

void quicksort(int A[], int F, int L)
{
    int pivotIndex;

    if (F < L) {
        partition(A, F, L, pivotIndex);
        quicksort(A, F, pivotIndex - 1);
        quicksort(A, pivotIndex + 1, L);
    }
}

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

void Swap(int& x, int& y)
{
    int temp = x;
    x = y;
    y = temp;
}
