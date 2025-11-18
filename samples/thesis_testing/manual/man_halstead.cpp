#include <iostream>
using namespace std; // Operators[4]: using,namespace,; | Operands[1]: std
int add(int a, int b) { // Operators[10]: int,int,int,{,(,),',' | Operands[4]: a,b,add
    return a + b;        // Operators[13]: +,return,; | Operands[6]: a,b 
}                        // Operators[14]: } | Operands[6]
int factorial(int n) {   // Operators[18]: int,int,{,(,) | Operands[8]: factorial,n
    int result = 1;      // Operators[21]: int,=,; | Operands[10]: result,1
    for(int i = 1; i <= n; ++i) { // Operators[29]: for,(,int,;,<=,;,++,),{ | Operands[15]: i,1,i,n,i
        result *= i;     // Operators[32]: *=,; | Operands[17]: result,i
    }                    // Operators[33]: } | Operands[17]
    return result;       // Operators[35]: return,; | Operands[18]: result
}                        // Operators[36]: }

int main () {            // Operators[40]: int,(,),{ | Operands[19]: main
    int x = 5;           // Operators[43]: int,=,; | Operands[21]: x, 5
    int y = 3;           // Operators[46]: int,=,; | Operands[23]: y, 3
    int z = add(x, y);   // Operators[54]: int,=,add,(,',',),; | Operands[26]: z,x,y
    cout << "Sum: " << z << endl; // Operators[60]: cout, <<, <<, <<, ; | Operands[29]: "Sum: ", z, endl  
    cout << "Factorial: " << factorial(x) << endl; 
    // Operators[68]: cout, <<, <<, factorial, (,), <<, ; | Operands[32]: "Factorial: ", x, endl
    return 0;            // Operators [71]: return, ; | Operands[33]: 0
} // Operators [72]: } | Operands [33]