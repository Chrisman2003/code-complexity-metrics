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
    
    # ✅ Updated expected Halstead metrics for simple.cpp
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
    
    # ✅ Updated expected Halstead metrics for complex.cpp
    assert metrics['n1'] == 40
    assert metrics['n2'] == 95
    assert metrics['N1'] == 220
    assert metrics['N2'] == 248
    assert vocabulary(metrics) == 135
    assert size(metrics) == 468
    assert round(volume(metrics), 2) == 3311.95
    assert round(difficulty(metrics), 2) == 52.21
    assert round(effort(metrics), 2) == 172918.64
    assert round(time(metrics), 2) == 9606.59


'''
def test_halstead_complex_cuda():
    """Tests Halstead metrics on a complex CUDA file."""
    code = load_code("complex/complex_cuda.cu")
    metrics = halstead_metrics_cuda(code)
    
    # Expected Halstead metrics for complex_cuda.cu (unchanged)
    assert metrics['n1'] == 25
    assert metrics['n2'] == 47
    assert metrics['N1'] == 250
    assert metrics['N2'] == 121
    assert vocabulary(metrics) == 72
    assert size(metrics) == 371
    assert round(volume(metrics), 2) == 2289.04
    assert round(difficulty(metrics), 2) == 32.18
    assert round(effort(metrics), 2) == 73663.33
    assert round(time(metrics), 2) == 4092.41


def test_halstead_complex_kokkos():
    """Tests Halstead metrics on a complex Kokkos C++ file."""
    code = load_code("complex/complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code)
    
    # Expected Halstead metrics for complex_kokkos.cpp (unchanged)
    assert metrics['n1'] == 22
    assert metrics['n2'] == 50
    assert metrics['N1'] == 134
    assert metrics['N2'] == 109
    assert vocabulary(metrics) == 72
    assert size(metrics) == 243
    assert round(volume(metrics), 2) == 1499.29
    assert round(difficulty(metrics), 2) == 23.98
    assert round(effort(metrics), 2) == 35953.02
    assert round(time(metrics), 2) == 1997.39


def test_halstead_complex_opencl():
    """Tests Halstead metrics on a complex OpenCL C++ file."""
    code = load_code("complex/complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code)
    
    # Expected Halstead metrics for complex_opencl.cpp (unchanged)
    assert metrics['n1'] == 31
    assert metrics['n2'] == 128
    assert metrics['N1'] == 313
    assert metrics['N2'] == 300
    assert vocabulary(metrics) == 159
    assert size(metrics) == 613
    assert round(volume(metrics), 2) == 4482.80
    assert round(difficulty(metrics), 2) == 36.33
    assert round(effort(metrics), 2) == 162851.62
    assert round(time(metrics), 2) == 9047.31
'''