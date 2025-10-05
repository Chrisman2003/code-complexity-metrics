#include <iostream>

// Function 1: simple if-else
int max(int a, int b) {
    if (a > b) return a;
    else return b;
}

// Function 2: loop with a decision
int sumEven(int n) {
    int sum = 0;
    for (int i = 1; i <= n; ++i) {
        if (i % 2 == 0) sum += i;
    }
    return sum;
}

// Function 3: another simple if
bool isPositive(int x) {
    if (x > 0) return true;
    else return false;
}

int main() {
    std::cout << max(5, 10) << "\n";
    std::cout << sumEven(10) << "\n";
    std::cout << isPositive(-3) << "\n";
    return 0;
}
