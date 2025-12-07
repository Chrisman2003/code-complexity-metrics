#include <iostream>

double fibonacci(int n) {
    if (n <= 1) return n;
    
    return fibonacci(n - 1) + fibonacci(n - 2);
}

double power (double x, int n) {
    if (n == 0) return 1.0;
    if (n < 0) return 1.0 / power(x, -n);
    double half = power(x, n / 2);
    if (n % 2 == 0) return half * half;
    return x * half * half;
}

