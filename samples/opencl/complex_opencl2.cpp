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
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if(err != CL_SUCCESS) { std::cerr << "Platform error\n"; return 1; }

    cl_device_id device;
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    if(err != CL_SUCCESS) { std::cerr << "Device error\n"; return 1; }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) { std::cerr << "Context error\n"; return 1; }

    // Use the newer command queue creation for Clang compatibility
    cl_command_queue_properties props[] = {0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    if(err != CL_SUCCESS) { std::cerr << "Queue error\n"; return 1; }

    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_a.data(), &err);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_b.data(), &err);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*sizeof(float), nullptr, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "add_vector", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, n*sizeof(float), h_c.data(), 0, nullptr, nullptr);

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
