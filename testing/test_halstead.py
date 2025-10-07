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
    assert metrics['n1'] == 35
    assert metrics['n2'] == 26
    assert metrics['N1'] == 263
    assert metrics['N2'] == 90
    assert vocabulary(metrics) == 61
    assert size(metrics) == 353
    assert round(volume(metrics), 2) == 2093.55
    assert round(difficulty(metrics), 2) == 60.58
    assert round(effort(metrics), 2) == 126820.83
    assert round(time(metrics), 2) == 7045.60


def test_halstead_complex_kokkos():
    """Tests Halstead metrics on a complex Kokkos C++ file."""
    code = load_code("complex/complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 21
    assert metrics['n2'] == 22
    assert metrics['N1'] == 133
    assert metrics['N2'] == 54
    assert vocabulary(metrics) == 43
    assert size(metrics) == 187
    assert round(volume(metrics), 2) == 1014.71
    assert round(difficulty(metrics), 2) == 25.77
    assert round(effort(metrics), 2) == 26151.88
    assert round(time(metrics), 2) == 1452.88



def test_halstead_complex_opencl():
    """Tests Halstead metrics on a complex OpenCL C++ file."""
    code = load_code("complex/complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code)  # Language-specific constructs analyzed!
    # Updated expected Halstead metrics based on latest analysis
    assert metrics['n1'] == 41
    assert metrics['n2'] == 62
    assert metrics['N1'] == 316
    assert metrics['N2'] == 164
    assert vocabulary(metrics) == 103
    assert size(metrics) == 480
    assert round(volume(metrics), 2) == 3209.52
    assert round(difficulty(metrics), 2) == 54.23
    assert round(effort(metrics), 2) == 174038.82
    assert round(time(metrics), 2) == 9668.82

'''
Edge Case:
Boundary Word Wrapping for keywords searched by the Halstead Function ?
'''