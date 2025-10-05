#include <iostream>

double fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

double power(double x, int n) {
    if (n == 0) return 1.0;
    if (n < 0) return 1.0 / power(x, -n);
    double half = power(x, n / 2);
    if (n % 2 == 0) return half * half;
    return x * half * half;
}


/*
int main() {
// Instruction sequence to demonstrate how instrucytions are grouped into singular nodes
    int x = 5;
    int y = 10;
    int z = 15;
    int result = 0; 
    x = y + z; // This line is grouped with the next one
    result = x * 2; // This line is grouped with the previous one
    return result; // This line is grouped with the previous two
    // The above lines should be grouped into a single node in the CFG
    // This is a simple example to illustrate how multiple instructions can be grouped together
}
*/