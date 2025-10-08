#include <iostream>
using namespace std;

void f(int x) {
    switch    (x) {
        case 1:
            foo();
        case 2:
            bar();
            fly();
        case 3:
            baz();
            break;
        default:
            qux();
            jni();
            bkqer();
    }
}

/*
int main() {
    int x = 2;
    switch (x) {
    case 1: 
        cout << "One\n";
    case 2: 
        cout << "Two\n"; 
        break;  // <-- This line would be missed
    case 3: 
        cout << "Three\n"; 
        break;
    default: 
        cout << "Default\n"; 
        break;
    }
    return 0;
    
}
*/