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

    # Updated expected values based on latest analysis (2025-09-28)
    assert metrics['n1'] == 19
    assert metrics['n2'] == 20
    assert metrics['N1'] == 54
    assert metrics['N2'] == 36
    assert vocabulary(metrics) == 39
    assert size(metrics) == 90
    assert round(volume(metrics), 2) == 475.69
    assert round(difficulty(metrics), 2) == 17.10
    assert round(effort(metrics), 2) == 8134.23
    assert round(time(metrics), 2) == 451.90


def test_halstead_complex_cpp():
    """Tests Halstead metrics on a more complex C++ file."""
    code = load_code("complex/complex.cpp")
    metrics = halstead_metrics_cpp(code)
    assert metrics['n1'] == 53
    assert metrics['n2'] == 99
    assert metrics['N1'] == 843
    assert metrics['N2'] == 548
    assert vocabulary(metrics) == 152
    assert size(metrics) == 1391
    assert round(volume(metrics), 2) == 10081.87
    assert round(difficulty(metrics), 2) == 146.69
    assert round(effort(metrics), 2) == 1478877.53
    assert round(time(metrics), 2) == 82159.86


def test_halstead_complex_cuda():
    """Tests Halstead metrics on a complex CUDA file."""
    code = load_code("complex/complex_cuda.cu")
    metrics = halstead_metrics_cuda(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 32
    assert metrics['n2'] == 29
    assert metrics['N1'] == 260
    assert metrics['N2'] == 93
    assert vocabulary(metrics) == 61
    assert size(metrics) == 353
    assert round(volume(metrics), 2) == 2093.55
    assert round(difficulty(metrics), 2) == 51.31
    assert round(effort(metrics), 2) == 107420.79
    assert round(time(metrics), 2) == 5967.82


def test_halstead_complex_kokkos():
    """Tests Halstead metrics on a complex Kokkos C++ file."""
    code = load_code("complex/complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 20
    assert metrics['n2'] == 23
    assert metrics['N1'] == 132
    assert metrics['N2'] == 55
    assert vocabulary(metrics) == 43
    assert size(metrics) == 187
    assert round(volume(metrics), 2) == 1014.71
    assert round(difficulty(metrics), 2) == 23.91
    assert round(effort(metrics), 2) == 24264.84
    assert round(time(metrics), 2) == 1348.05


def test_halstead_complex_opencl():
    """Tests Halstead metrics on a complex OpenCL C++ file."""
    code = load_code("complex/complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code)  # Language-specific constructs analyzed!
    # Updated expected Halstead metrics for complex_opencl.cpp
    assert metrics['n1'] == 39
    assert metrics['n2'] == 64
    assert metrics['N1'] == 314
    assert metrics['N2'] == 166
    assert vocabulary(metrics) == 103
    assert size(metrics) == 480
    assert round(volume(metrics), 2) == 3209.52
    assert round(difficulty(metrics), 2) == 50.58
    assert round(effort(metrics), 2) == 162331.52
    assert round(time(metrics), 2) == 9018.42

'''
Edge Case:
Boundary Word Wrapping for keywords searched by the Halstead Function ?
'''