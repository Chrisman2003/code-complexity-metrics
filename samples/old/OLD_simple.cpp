#include <iostream>
using namespace std;

int main() {
    int x = 5;
    int y = 10;
    x = y + 2;
    y = x * 3;

    if (x < y) {
        cout << "x is less than y" << endl;
    } else {
        cout << "x is greater or equal to y" << endl;
    }

    for (int i = 0; i < 5; i++) {
        x += i;
    }

    return 0;
}
