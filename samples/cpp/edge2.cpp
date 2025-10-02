#include <iostream>
using namespace std;

int main() {
    int a = 5;
    int b = 10;
    int c = 15;

    // Boolean operators in actual code
    if ((a < b) && (b < c)) {   // && should be counted
        cout << "a < b && b < c" << endl;
    }

    if ((a > 0) || (c < 20)) {  // || should be counted
        cout << "a > 0 || c < 20" << endl;
    }

    // Boolean operators inside strings (should NOT be counted)
    cout << "this && should not count" << endl;
    cout << "this || should not count" << endl;
    cout << "a or b in string" << endl;

    // Boolean operators inside comments (should NOT be counted)
    // if (a < b && b < c) { ... } 
    /* c > b || a < c */

    // Mixed case: conditional operator
    int result = (a < b) ? b : a;  // ? should be counted

    // Edge cases with multiple operators in one line
    if ((a < b && b < c) || (c > a && a != 0)) {
        cout << "complex condition with multiple operators" << endl;
    }

    return 0;
}
