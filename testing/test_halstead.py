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
    code = load_code("cpp/OLD_simple.cpp")
    metrics = halstead_metrics_cpp(code)
    # Updated expected values based on latest analysis
    assert metrics['n1'] == 23
    assert metrics['n2'] == 16
    assert metrics['N1'] == 58
    assert metrics['N2'] == 32
    assert vocabulary(metrics) == 39
    assert size(metrics) == 90
    assert round(volume(metrics), 2) == 475.69
    assert round(difficulty(metrics), 2) == 23.00
    assert round(effort(metrics), 2) == 10940.78
    assert round(time(metrics), 2) == 607.82


def test_halstead_complex_cpp():
    """Tests Halstead metrics on a more complex C++ file."""
    code = load_code("complex/complex.cpp")
    metrics = halstead_metrics_cpp(code)
    assert metrics['n1'] == 59
    assert metrics['n2'] == 95
    assert metrics['N1'] == 914
    assert metrics['N2'] == 503
    assert vocabulary(metrics) == 154
    assert size(metrics) == 1417
    assert round(volume(metrics), 2) == 10297.04
    assert round(difficulty(metrics), 2) == 156.19
    assert round(effort(metrics), 2) == 1608342.91
    assert round(time(metrics), 2) == 89352.38


def test_halstead_complex_cuda():
    """Tests Halstead metrics on a complex CUDA file."""
    code = load_code("complex/complex_cuda.cu")
    metrics = halstead_metrics_cuda(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 39
    assert metrics['n2'] == 22
    assert metrics['N1'] == 282
    assert metrics['N2'] == 81
    assert vocabulary(metrics) == 61
    assert size(metrics) == 363
    assert round(volume(metrics), 2) == 2152.86
    assert round(difficulty(metrics), 2) == 71.80
    assert round(effort(metrics), 2) == 154565.39
    assert round(time(metrics), 2) == 8586.97


def test_halstead_complex_kokkos():
    """Tests Halstead metrics on a complex Kokkos C++ file."""
    code = load_code("complex/complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 26
    assert metrics['n2'] == 22
    assert metrics['N1'] == 141
    assert metrics['N2'] == 54
    assert vocabulary(metrics) == 48
    assert size(metrics) == 195
    assert round(volume(metrics), 2) == 1089.07
    assert round(difficulty(metrics), 2) == 31.91
    assert round(effort(metrics), 2) == 34751.16
    assert round(time(metrics), 2) == 1930.62


def test_halstead_complex_opencl():
    """Tests Halstead metrics on a complex OpenCL C++ file."""
    code = load_code("complex/complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code)  # Language-specific constructs analyzed!
    # Updated expected Halstead metrics based on latest analysis
    assert metrics['n1'] == 52
    assert metrics['n2'] == 51
    assert metrics['N1'] == 339
    assert metrics['N2'] == 153
    assert vocabulary(metrics) == 103
    assert size(metrics) == 492
    assert round(volume(metrics), 2) == 3289.76
    assert round(difficulty(metrics), 2) == 78.00
    assert round(effort(metrics), 2) == 256601.14
    assert round(time(metrics), 2) == 14255.62

'''
Edge Case:
Boundary Word Wrapping for keywords searched by the Halstead Function ?
'''