#include <iostream>
#include <vector>

// Partition function
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high]; // choose last element as pivot
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Quicksort recursive function
void quicksort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

// Helper to print array
void printArray(const std::vector<int>& arr) {
    for (int x : arr) std::cout << x << " ";
    std::cout << "\n";
}

// Main function
int main() {
    std::vector<int> data = {9, 3, 1, 5, 13, 12, 10, 7};
    std::cout << "Original array:\n";
    printArray(data);

    quicksort(data, 0, data.size() - 1);

    std::cout << "Sorted array:\n";
    printArray(data);
    return 0;
}
