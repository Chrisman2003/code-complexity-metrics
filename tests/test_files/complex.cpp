#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return false;
    }
    return true;
}

int sumVector(const vector<int>& v) {
    int sum = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i];
    }
    return sum;
}

int main() {
    vector<int> numbers = {2, 3, 4, 5, 15, 17};
    int total = 0;

    for (auto num : numbers) {
        if (isPrime(num) && num < 10) {
            cout << num << " is prime and less than 10" << endl;
        } else if (!isPrime(num) || num == 17) {
            cout << num << " is either not prime or equal to 17" << endl;
        } else {
            cout << num << " did not match any condition" << endl;
        }
        total += num;
    }

    int vectorSum = sumVector(numbers);
    double ratio = (double)total / (double)vectorSum;

    cout << "Total sum: " << total << endl;
    cout << "Vector sum: " << vectorSum << endl;
    cout << "Ratio: " << ratio << endl;

    return 0;
}
