#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <iostream>
#include <vector>

const char* kernelSource = R"CLC(
__kernel void add_vector(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
)CLC";

int main() {
    int n = 1024;
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for(int i=0; i<n; i++){ h_a[i] = i; h_b[i] = i*i; }

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_a.data(), NULL);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_b.data(), NULL);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*sizeof(float), NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "add_vector", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, n*sizeof(float), h_c.data(), 0, NULL, NULL);

    std::cout << "OpenCL first element: " << h_c[0] << std::endl;

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

/*
Explanation:
This program demonstrates a simple OpenCL example that performs element-wise addition of two vectors.
1. Two host vectors h_a and h_b are initialized on the CPU.
2. OpenCL platform, device, context, and command queue are created.
3. Device buffers d_a, d_b, and d_c are allocated, and host data is copied to device memory.
4. A simple kernel "add_vector" is compiled and executed on the GPU, adding corresponding elements of a and b into c.
5. The result vector h_c is read back to the CPU and the first element is printed.
6. All OpenCL resources are released at the end.
*/