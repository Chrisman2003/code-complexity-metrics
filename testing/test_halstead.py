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
    code = load_code("old/OLD_simple.cpp")
    metrics = halstead_metrics_cpp(code)
    
    assert metrics['n1'] == 18
    assert metrics['n2'] == 24
    assert metrics['N1'] == 50
    assert metrics['N2'] == 43
    assert vocabulary(metrics) == 42
    assert size(metrics) == 93
    assert round(volume(metrics), 2) == 501.49
    assert round(difficulty(metrics), 2) == 16.12
    assert round(effort(metrics), 2) == 8086.45
    assert round(time(metrics), 2) == 449.25


def test_halstead_complex_cpp():
    """Tests Halstead metrics on a more complex C++ file."""
    code = load_code("complex/complex.cpp")
    metrics = halstead_metrics_cpp(code)
    
    assert metrics['n1'] == 55
    assert metrics['n2'] == 122
    assert metrics['N1'] == 1069
    assert metrics['N2'] == 595
    assert vocabulary(metrics) == 177
    assert size(metrics) == 1664
    assert round(volume(metrics), 2) == 12426.10
    assert round(difficulty(metrics), 2) == 134.12
    assert round(effort(metrics), 2) == 1666573.69
    assert round(time(metrics), 2) == 92587.43



def test_halstead_complex_cuda():
    """Tests Halstead metrics on a complex CUDA file."""
    code = load_code("complex/complex_cuda.cu")
    metrics = halstead_metrics_cuda(code) # Language specific constructs ANALYZED!
    # Expected Halstead metrics for complex_cuda.cu with cuda language extension
    assert metrics['n1'] == 32
    assert metrics['n2'] == 40
    assert metrics['N1'] == 263
    assert metrics['N2'] == 108
    assert vocabulary(metrics) == 72
    assert size(metrics) == 371
    assert round(volume(metrics), 2) == 2289.04
    assert round(difficulty(metrics), 2) == 43.20
    assert round(effort(metrics), 2) == 98886.62
    assert round(time(metrics), 2) == 5493.70


def test_halstead_complex_kokkos():
    """Tests Halstead metrics on a complex Kokkos C++ file."""
    code = load_code("complex/complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code) # Language specific constructs ANALYZED!
    # Expected Halstead metrics for complex_kokkos.cpp with Kokkos language extension
    assert metrics['n1'] == 24
    assert metrics['n2'] == 48
    assert metrics['N1'] == 150
    assert metrics['N2'] == 93
    assert vocabulary(metrics) == 72
    assert size(metrics) == 243
    assert round(volume(metrics), 2) == 1499.29
    assert round(difficulty(metrics), 2) == 23.25
    assert round(effort(metrics), 2) == 34858.53
    assert round(time(metrics), 2) == 1936.59


def test_halstead_complex_opencl():
    """Tests Halstead metrics on a complex OpenCL C++ file."""
    code = load_code("complex/complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code) # Language specific constructs ANALYZED!
    # Expected Halstead metrics for complex_opencl.cpp with OpenCL language extension
    assert metrics['n1'] == 41
    assert metrics['n2'] == 118
    assert metrics['N1'] == 342
    assert metrics['N2'] == 271
    assert vocabulary(metrics) == 159
    assert size(metrics) == 613
    assert round(volume(metrics), 2) == 4482.80
    assert round(difficulty(metrics), 2) == 47.08
    assert round(effort(metrics), 2) == 211052.37
    assert round(time(metrics), 2) == 11725.13
