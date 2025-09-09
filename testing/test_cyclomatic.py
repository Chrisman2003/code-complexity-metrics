import os
from code_complexity.metrics.cyclomatic import basic_compute_cyclomatic

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


def test_cyclomatic_example_py():
    """Tests cyclomatic complexity calculation on a simple Python file.

    The expected complexity is at least 1.
    """
    code = load_code("non_cpp/OLD_example.py")
    assert basic_compute_cyclomatic(code) >= 1


def test_cyclomatic_simple_cpp():
    """Tests cyclomatic complexity calculation on a simple C++ file.

    The expected complexity is at least 1.
    """
    code = load_code("simple/OLD_simple.cpp")
    assert basic_compute_cyclomatic(code) >= 1


def test_cyclomatic_complex_cpp():
    """Tests cyclomatic complexity calculation on a more complex C++ file.

    The expected complexity is at least 1.
    """
    code = load_code("complex/complex.cpp")
    assert basic_compute_cyclomatic(code) >= 1
