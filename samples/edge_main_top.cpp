#include <iostream>

/*
Code is trying to call the functions foo and bar within 
the main function before they have been declared.
In C++, all functions must be declared or defined before they are called. 
Since foo and bar are defined after main, the compiler doesnt know 
what they are when it encounters them on lines 3 and 4.
*/

int foo(int x);
int bar(int y);

int main() {
    std::cout << foo(5) << std::endl;
    std::cout << bar(2) << std::endl;
    return 0;
}


int foo(int x) {
    if (x > 0) {
        return x;
    } else {
        return -x;
    }
}

int bar(int y) {
    switch (y) {
        case 1: return 10;
        case 2: return foo(20);
        default: return 0;
    }
}


