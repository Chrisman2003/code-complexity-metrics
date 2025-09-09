#include <iostream>

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

int main() {
    std::cout << foo(5) << std::endl;
    std::cout << bar(2) << std::endl;
    return 0;
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