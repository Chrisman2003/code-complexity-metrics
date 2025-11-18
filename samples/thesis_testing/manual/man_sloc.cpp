#include <iostream> // Not counted
using namespace std; // +1

int main() { // +1
    int a = 5; // +1
    int b = 10; // +1

    // Boolean operators in actual code //
    if ((a < b) && (b < c)) {   // && should be counted // +1
        cout << "a < b && b < c" << endl; // +1
    } // +1
    /* 
    Not counted //
    */

    // Boolean operators inside strings (should NOT be counted)
    cout << 'single quote test' << endl; // +1
    cout << 'this || should not count' << endl; // +1
    // Mixed case: conditional operator
    int result = (a < b) ? b : a;  // ? should be counted // +1
 
    // Edge cases with multiple operators in one line //
    if ((a < b && b < c) || (c > a && a != 0)) { // +1
        cout << "complex condition with multiple operators" << endl; // +1
    } // +1
    return 0; // +1
} // +1