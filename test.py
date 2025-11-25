import re

Operators1 = {'+=', 'blockIdx', 'explicit', '[', 'for', 'else', '#', '-', 'cudaError_t', ')', 'default', 'operator', 'cudaMemcpyDeviceToHost', '==', 'size_t', '<=', 'continue', '*=', 'void', 'struct', 'true', 'unsigned', 'char', 'threadIdx', '__global__', ':', 'cudaMalloc', 'double', '=', 'false', ']', 'thrust::reduce', ';', 'typename', '>=', ',', 'throw', '%', '__device__', 'blockDim', '(', 'std::unique_ptr', 'sizeof', '<<', 'std::string', '/', '~', 'auto', 'if', '!', 'cudaGetErrorString', 'bool', '&', 'public', 'template', '>', '{', 'float3', 'using', 'nullptr', '!=', '?', 'int', '&=', 'int3', '}', 'cudaMemcpy', '<', 'cudaGetLastError', '.', 'thrust::device_ptr', 'cudaMemcpyHostToDevice', '::', '||', '+', 'override', 'cudaFree', 'class', '++', 'thrust::device_pointer_cast', '*', 'inline', 'cudaSuccess', 'static_cast', 'float4', 'std::vector', 'return', 'private', '&&', '|=', 'const', '>>'}
Operators2 = {'&', 'auto', 'cudaGetErrorString', 'float3', '#', '||', 'int3', 'nullptr', '__device__', 'threadIdx', 'struct', '%', 'bool', 'continue', '&=', 'else', 'cudaMemcpyDeviceToHost', '+', 'blockIdx', '*', '&&', 'cudaMemcpy', 'inline', '~', 'std::unique_ptr', '<=', ')', '.', '?', '!=', 'thrust::device_ptr', '}', 'class', '{', '|=', '+=', 'cudaSuccess', 'typename', '>=', 'cudaMalloc', 'char', 'private', '<<', 'template', '=', '-', 'copy', 'false', 'std::string', 'std::vector', 'default', 'using', '>', '!', 'throw', 'const', 'static_cast', ';', 'float4', 'override', 'size_t', 'void', 'if', 'true', '__global__', '/', 'double', 'cudaFree', '==', 'cudaGetLastError', 'cudaMemcpyHostToDevice', '*=', ',', 'thrust::device_pointer_cast', '(', 'cudaError_t', 'unsigned', 'for', 'thrust::reduce', ']', '++', ':', 'public', 'int', 'sizeof', 'operator', '[', 'return', 'blockDim', 'acc_pot', '<', 'explicit', '::', '>>'}
print(Operators2 - Operators1)

def detect_parallel_framework(code: str) -> str:
    """
    Automatically detect the parallelizing framework used in a source file.
    Returns one or a multiple of ['cpp', 'cuda', 'opencl', 'kokkos', 'openmp', 
                    'adaptivecpp', 'openacc', 'opengl_vulkan', 
                    'webgpu', 'boost', 'metal', 'thrust']
    """
    lib_patterns = { # Assuming correct library declarations
        "cuda": [r'#include\s*<cuda'],
        "opencl": [r'#include\s*<CL/cl'],
        "kokkos": [r'#include\s*<Kokkos'],
        "openmp": [r'#include\s*[<"]omp'],
        "adaptivecpp": [r'#include\s*<CL/sycl'],
        "openacc": [r'#include\s*<openacc'],
        "opengl_vulkan": [r'#include\s*<vulkan'],
        "webgpu": [r'#include\s*<wgpu'],
        "boost": [r'#include\s*"boost'],
        "metal": [r'#include\s*<Metal'],
        "thrust": [r'#include\s*<thrust'],
    }
    detected_languages = {"cpp"}
    for lang, patterns in lib_patterns.items():
        matches = re.findall(patterns[0], code)
        if len(matches) > 0:
            detected_languages.add(lang)
    return detected_languages

# -------------------------------
# Test cases
# -------------------------------
def main():
    tests = [
        ("#include <cuda_runtime.h>", {"cpp", "cuda"}),
        ("#include <CL/cl.hpp>", {"cpp", "opencl"}),
        ("#include <CL/sycl.hpp>", {"cpp", "adaptivecpp"}),
        ("#include <openacc.h>", {"cpp", "openacc"}),
        ("#include <omp.h>", {"cpp", "openmp"}),

        ("#include <cuda_runtime.h>\n#include <CL/sycl.hpp>", {"cpp", "cuda", "adaptivecpp"}),
        ("#include <CL/cl.hpp>\n#include <openacc.h>", {"cpp", "opencl", "openacc"}),

        ("#include <cuda_runtime.h>\n#include <vulkan/vulkan.hpp>", {"cpp", "cuda", "opengl_vulkan"}),

        ("#include <iostream>\nint main() { return 0; }", {"cpp"}),
]

    for i, (code, expected) in enumerate(tests, 1):
        detected = detect_parallel_framework(code)
        assert detected == expected, f"Test {i} failed: detected {detected}, expected {expected}"
        print(f"Test {i} passed: {detected}")

if __name__ == "__main__":
    main()