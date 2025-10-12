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
    assert metrics['n1'] == 22
    assert metrics['n2'] == 14
    assert metrics['N1'] == 57
    assert metrics['N2'] == 30
    assert vocabulary(metrics) == 36
    assert size(metrics) == 87
    assert round(volume(metrics), 2) == 449.78
    assert round(difficulty(metrics), 2) == 23.57
    assert round(effort(metrics), 2) == 10602.04
    assert round(time(metrics), 2) == 589.00
    

def test_halstead_complex_cpp():
    """Tests Halstead metrics on a more complex C++ file."""
    code = load_code("complex/complex.cpp")
    metrics = halstead_metrics_cpp(code)
    assert metrics['n1'] == 59
    assert metrics['n2'] == 88
    assert metrics['N1'] == 884
    assert metrics['N2'] == 496
    assert vocabulary(metrics) == 147
    assert size(metrics) == 1380
    assert round(volume(metrics), 2) == 9935.55
    assert round(difficulty(metrics), 2) == 166.27
    assert round(effort(metrics), 2) == 1652010.64
    assert round(time(metrics), 2) == 91778.37


def test_halstead_complex_cuda():
    """Tests Halstead metrics on a complex CUDA file."""
    code = load_code("complex/complex_cuda.cu")
    metrics = halstead_metrics_cuda(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 38
    assert metrics['n2'] == 19
    assert metrics['N1'] == 274
    assert metrics['N2'] == 76
    assert vocabulary(metrics) == 57
    assert size(metrics) == 350
    assert round(volume(metrics), 2) == 2041.51
    assert round(difficulty(metrics), 2) == 76.00
    assert round(effort(metrics), 2) == 155154.87
    assert round(time(metrics), 2) == 8619.72


def test_halstead_complex_kokkos():
    """Tests Halstead metrics on a complex Kokkos C++ file."""
    code = load_code("complex/complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code)  # Language-specific constructs ANALYZED!
    assert metrics['n1'] == 24
    assert metrics['n2'] == 19
    assert metrics['N1'] == 126
    assert metrics['N2'] == 63
    assert vocabulary(metrics) == 43
    assert size(metrics) == 189
    assert round(volume(metrics), 2) == 1025.56
    assert round(difficulty(metrics), 2) == 39.79
    assert round(effort(metrics), 2) == 40806.65
    assert round(time(metrics), 2) == 2267.04
    

def test_halstead_complex_opencl():
    """Tests Halstead metrics on a complex OpenCL C++ file."""
    code = load_code("complex/complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code)  # Language-specific constructs analyzed!
    # Updated expected Halstead metrics based on latest analysis
    assert metrics['n1'] == 52
    assert metrics['n2'] == 47
    assert metrics['N1'] == 328
    assert metrics['N2'] == 147
    assert vocabulary(metrics) == 99
    assert size(metrics) == 475
    assert round(volume(metrics), 2) == 3148.94
    assert round(difficulty(metrics), 2) == 81.32
    assert round(effort(metrics), 2) == 256069.48
    assert round(time(metrics), 2) == 14226.08
    
'''
Edge Case:
Boundary Word Wrapping for keywords searched by the Halstead Function ?
'''