import os
from code_complexity.metrics.cyclomatic import *

# Directory containing test files for code complexity analysis
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
TEST_FILES_DIR = os.path.abspath(TEST_FILES_DIR)  # Absolute path for consistency

def load_code(filename):
    """Loads the content of a code file.

    Args:
        filename (str): Name of the file to load from TEST_FILES_DIR.

    Returns:
        str: The content of the file as a string.
    """
    with open(os.path.join(TEST_FILES_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()


def test_cyclomatic_simple_cpp():
    """
    Tests cyclomatic complexity calculation on a simple C++ File.
    """
    code = load_code("cpp/OLD_simple.cpp")
    assert basic_compute_cyclomatic(code) == 4
    assert compute_cyclomatic(code, "OLD_simple.cpp") == 4


def test_cyclomatic_edge_cpp():
    """
    Tests cyclomatic complexity for specified edge case File.
    """
    code = load_code("cpp/edge.cpp")
    assert basic_compute_cyclomatic(code) == 4
    assert compute_cyclomatic(code, "edge.cpp") == 6


def test_cyclomatic_complex_cpp():
    """
    Tests cyclomatic complexity calculation on a more complex C++ File.
    """
    code = load_code("complex/complex.cpp")
    assert basic_compute_cyclomatic(code) == 50
    assert compute_cyclomatic(code, "complex.cpp") == 52

    
def test_cyclomatic_hyper_complex_cpp():
    """
    Tests cyclomatic complexity calculation on a very complex C++ File.
    """
    code = load_code("complex/hyper_complex.cpp")
    assert basic_compute_cyclomatic(code) == 83
    assert compute_cyclomatic(code, "hyper_complex.cpp") == 83