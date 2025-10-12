import os
from code_complexity.metrics.sloc import compute_sloc

# Directory containing test files for SLOC calculation
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


def test_sloc_simple_cpp():
    """Tests the source lines of code (SLOC) calculation on a simple C++ file."""
    code = load_code("cpp/OLD_simple.cpp")
    # Expected SLOC for OLD_simple.cpp
    assert compute_sloc(code) == 17


def test_cyclomatic_edge_cpp():
    """
    Tests cyclomatic complexity for specified edge case File.
    """
    code = load_code("cpp/edge.cpp")
    assert compute_sloc(code) == 11
 

def test_sloc_complex_cpp():
    """Tests the source lines of code (SLOC) calculation on a complex C++ file."""
    code = load_code("complex/complex.cpp")
    
    # Expected SLOC for complex.cpp
    assert compute_sloc(code) == 158

def test_cyclomatic_hyper_complex_cpp():
    """
    Tests cyclomatic complexity calculation on a very complex C++ File.
    """
    code = load_code("complex/hyper_complex.cpp")
    assert compute_sloc(code) == 206