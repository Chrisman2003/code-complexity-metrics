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
    //Hello There
    return true;
}

int sumVector(const vector<int>& v) {
    int sum = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i];
    }
    /*
    This is 
    a great place
    to put a 
    comment
    */
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

/*
Explanation:
1. The program defines a function isPrime(n) to check if a number is prime.
2. sumVector(v) calculates the sum of all elements in a vector.
3. In main(), a vector of numbers is initialized.
4. Each number is tested:
   - If it is prime and less than 10, a message is printed.
   - If it is not prime or equal to 17, a different message is printed.
   - Otherwise, a fallback message is printed.
5. The total sum of numbers is computed in the loop.
6. The sumVector function also calculates the sum for comparison.
7. The ratio of the loop total to the function sum is printed.
8. The program demonstrates loops, conditionals, vector usage, type casting, and functions.
*/
