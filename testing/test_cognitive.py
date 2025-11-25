# -----------------------------------------------------------------------------
# Cognitive Complexity Tests for C++ Source Files
# -----------------------------------------------------------------------------
# Parametrized pytest module to verify cognitive complexity
# computation for multiple C++ sample files. Reports mismatches between
# detected and expected values.
# -----------------------------------------------------------------------------
import os
import pytest
from code_complexity.metrics.utils import load_code
from code_complexity.metrics.cognitive import regex_compute_cognitive

# Directory containing test files for code complexity analysis
TEST_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

# -------------------------------
# Test cases: (filename, expected cognitive complexity)
# -------------------------------
test_cases = [
    ("cpp/OLD_simple.cpp", 3),
    ("cpp/edge.cpp", 4),
    ("cpp/complex.cpp", 38),
    ("cpp/hyper_complex.cpp", 90),
]

@pytest.mark.parametrize("filename,expected", test_cases)
def test_cognitive_complexity(filename, expected):
    """
    Test cognitive complexity computation for multiple C++ source files.

    This is a parametrized pytest function that checks whether the
    `regex_compute_cognitive` function correctly computes the cognitive
    complexity for a set of test files.

    Args:
        filename (str): Relative path to the C++ source file under test.
        expected (int): Expected cognitive complexity value for the file.

    Raises:
        AssertionError: If the computed cognitive complexity does not match
                        the expected value.
    """
    code = load_code(filename, TEST_FILES_DIR)
    detected = regex_compute_cognitive(code)
    assert detected == expected, f"File {filename}: detected {detected}, expected {expected}"
