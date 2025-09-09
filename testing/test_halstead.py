import os
from code_complexity.metrics.halstead import *

# Directory containing test files for Halstead metrics
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
TEST_FILES_DIR = os.path.abspath(TEST_FILES_DIR)


def load_code(filename):
    """Loads the content of a code file.

    Args:
        filename (str): Name of the file to load from TEST_FILES_DIR.

    Returns:
        str: The content of the file as a string.
    """
    with open(os.path.join(TEST_FILES_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()


def test_halstead_simple_cpp():
    """Tests Halstead metrics on a simple C++ file."""
    code = load_code("simple/OLD_simple.cpp")
    metrics = halstead_metrics_cpp(code)
    
    # Expected Halstead metrics for simple.cpp
    assert metrics['n1'] == 21
    assert metrics['n2'] == 21
    assert metrics['N1'] == 53
    assert metrics['N2'] == 40
    assert vocabulary(metrics) == 42
    assert size(metrics) == 93
    assert round(volume(metrics), 2) == 501.49
    assert round(difficulty(metrics), 2) == 20.00
    assert round(effort(metrics), 2) == 10029.71
    assert round(time(metrics), 2) == 557.21


def test_halstead_complex_cpp():
    """Tests Halstead metrics on a more complex C++ file."""
    code = load_code("complex/complex.cpp")
    metrics = halstead_metrics_cpp(code)
    
    # Expected Halstead metrics for complex.cpp
    assert metrics['n1'] == 39
    assert metrics['n2'] == 96
    assert metrics['N1'] == 227
    assert metrics['N2'] == 241
    assert vocabulary(metrics) == 135
    assert size(metrics) == 468
    assert round(volume(metrics), 2) == 3311.95
    assert round(difficulty(metrics), 2) == 48.95
    assert round(effort(metrics), 2) == 162130.29
    assert round(time(metrics), 2) == 9007.24


def test_halstead_complex_cuda():
    """Tests Halstead metrics on a complex CUDA file."""
    code = load_code("complex/complex_cuda.cu")
    metrics = halstead_metrics_cuda(code)
    
    # Expected Halstead metrics for complex_cuda.cu
    assert metrics['n1'] == 34
    assert metrics['n2'] == 38
    assert metrics['N1'] == 258
    assert metrics['N2'] == 113
    assert vocabulary(metrics) == 72
    assert size(metrics) == 371
    assert round(volume(metrics), 2) == 2289.04
    assert round(difficulty(metrics), 2) == 50.55
    assert round(effort(metrics), 2) == 115717.11
    assert round(time(metrics), 2) == 6428.73


def test_halstead_complex_kokkos():
    """Tests Halstead metrics on a complex Kokkos C++ file."""
    code = load_code("complex/complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code)
    
    # Expected Halstead metrics for complex_kokkos.cpp
    assert metrics['n1'] == 26
    assert metrics['n2'] == 46
    assert metrics['N1'] == 153
    assert metrics['N2'] == 90
    assert vocabulary(metrics) == 72
    assert size(metrics) == 243
    assert round(volume(metrics), 2) == 1499.29
    assert round(difficulty(metrics), 2) == 25.43
    assert round(effort(metrics), 2) == 38134.16
    assert round(time(metrics), 2) == 2118.56


def test_halstead_complex_opencl():
    """Tests Halstead metrics on a complex OpenCL C++ file."""
    code = load_code("complex/complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code)
    
    # Expected Halstead metrics for complex_opencl.cpp
    assert metrics['n1'] == 39
    assert metrics['n2'] == 119
    assert metrics['N1'] == 333
    assert metrics['N2'] == 280
    assert vocabulary(metrics) == 158
    assert size(metrics) == 613
    assert round(volume(metrics), 2) == 4477.22
    assert round(difficulty(metrics), 2) == 45.88
    assert round(effort(metrics), 2) == 205425.28
    assert round(time(metrics), 2) == 11412.52
