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
    assert metrics['n2'] == 13
    assert metrics['N1'] == 58
    assert metrics['N2'] == 29
    assert vocabulary(metrics) == 36
    assert size(metrics) == 87
    assert round(volume(metrics), 2) == 449.78
    assert round(difficulty(metrics), 2) == 25.65
    assert round(effort(metrics), 2) == 11538.68
    assert round(time(metrics), 2) == 641.04
    

def test_halstead_complex_cpp():
    """Tests Halstead metrics on a more complex C++ file."""
    code = load_code("complex/complex.cpp")
    metrics = halstead_metrics_cpp(code)
    assert metrics['n1'] == 60
    assert metrics['n2'] == 87
    assert metrics['N1'] == 885
    assert metrics['N2'] == 495
    assert vocabulary(metrics) == 147
    assert size(metrics) == 1380
    assert round(volume(metrics), 2) == 9935.55
    assert round(difficulty(metrics), 2) == 170.69
    assert round(effort(metrics), 2) == 1695895.23
    assert round(time(metrics), 2) == 94216.40


def test_halstead_complex_cuda():
    """Tests Halstead metrics on a complex CUDA file."""
    code = load_code("complex/complex_cuda.cu")
    metrics = halstead_metrics_cuda(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 39
    assert metrics['n2'] == 18
    assert metrics['N1'] == 275
    assert metrics['N2'] == 75
    assert vocabulary(metrics) == 57
    assert size(metrics) == 350
    assert round(volume(metrics), 2) == 2041.51
    assert round(difficulty(metrics), 2) == 81.25
    assert round(effort(metrics), 2) == 165872.81
    assert round(time(metrics), 2) == 9215.16


def test_halstead_complex_kokkos():
    """Tests Halstead metrics on a complex Kokkos C++ file."""
    code = load_code("complex/complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 25
    assert metrics['n2'] == 18
    assert metrics['N1'] == 127
    assert metrics['N2'] == 62
    assert vocabulary(metrics) == 43
    assert size(metrics) == 189
    assert round(volume(metrics), 2) == 1025.56
    assert round(difficulty(metrics), 2) == 43.06
    assert round(effort(metrics), 2) == 44156.23
    assert round(time(metrics), 2) == 2453.12
    
    
def test_halstead_complex_opencl():
    """Tests Halstead metrics on a complex OpenCL C++ file."""
    code = load_code("complex/complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code)  # Language-specific constructs analyzed!
    # Updated expected Halstead metrics based on latest analysis
    assert metrics['n1'] == 53
    assert metrics['n2'] == 46
    assert metrics['N1'] == 329
    assert metrics['N2'] == 146
    assert vocabulary(metrics) == 99
    assert size(metrics) == 475
    assert round(volume(metrics), 2) == 3148.94
    assert round(difficulty(metrics), 2) == 84.11
    assert round(effort(metrics), 2) == 264853.61
    assert round(time(metrics), 2) == 14714.09

    
'''
Edge Case:
Boundary Word Wrapping for keywords searched by the Halstead Function ?
'''