#include <CL/cl.h>
#include <iostream>

const char* kernelSource1 = "__kernel void add(__global float* a, __global float* b, __global float* c) { int id = get_global_id(0); c[id] = a[id] + b[id]; }";

const char* notAKernel = "This is just a string with no kernel keyword, should be stripped";

const char* trickyEscapes = "String with \\\"escaped quotes\\\" and __kernel inside should be kept";

const char* rawKernel = R"CLC(
__kernel void multiply(__global float* a, __global float* b, __global float* c) {
    int id = get_global_id(0);
    c[id] = a[id] * b[id];
}
)CLC";

const char* rawNonKernel = R"TXT(
This is just a raw string with no __kernel, should be removed entirely.
)TXT";

int main() {
    char c = 'x';  // character literal should be stripped
    char newline = '\n'; // escape sequence char literal should be stripped

    std::cout << "Host message: before kernel" << std::endl;  // should be stripped
    std::cout << "__kernel inside string but not code" << std::endl; // should be kept (contains __kernel)

    // Fake comments with __kernel should NOT trigger keeping anything
    // __kernel void bogus() { should not be counted };

    // Multi-line normal string kernel
    const char* kernelSource2 = "__kernel void subtract(\n"
                                "__global float* a,\n"
                                "__global float* b,\n"
                                "__global float* c) {\n"
                                "  int id = get_global_id(0);\n"
                                "  c[id] = a[id] - b[id];\n"
                                "}\n";

    // Multiple kernels back-to-back
    const char* multipleKernels = "__kernel void k1() { } __kernel void k2() { }";

    return 0;
}
