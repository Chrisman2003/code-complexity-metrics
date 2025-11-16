import pytest
from code_complexity.metrics.halstead import detect_parallel_framework

# -------------------------------
# Test cases
# -------------------------------
test_cases = [
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

@pytest.mark.parametrize("code,expected", test_cases)
def test_detect_framework(code, expected):
    detected = detect_parallel_framework(code)
    assert detected == expected, f"Detected: {detected}, Expected: {expected}"