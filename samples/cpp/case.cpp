#include <iostream>
using namespace std;

int main() {
    int x = 2;
    switch (x) {
    case 1: cout << "One\n"; break;
          case 2: cout << "Two\n"; break;  // <-- This line would be missed
           case 3: cout << "Three\n"; break;
    default: cout << "Default\n"; break;
    }
    return 0;
}
