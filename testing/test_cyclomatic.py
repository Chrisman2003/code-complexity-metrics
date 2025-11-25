# -----------------------------------------------------------------------------
# Cyclomatic Complexity Tests for C++ Source Files
# -----------------------------------------------------------------------------
# Parametrized pytest module to verify cyclomatic complexity
# computation for multiple C++ sample files. Reports mismatches between
# detected and expected values.
# -----------------------------------------------------------------------------
import os
import pytest
from code_complexity.metrics.cyclomatic import regex_compute_cyclomatic, cfg_compute_cyclomatic
from code_complexity.metrics.utils import load_code

# Directory containing test files for code complexity analysis - Using absolute path
TEST_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

# -------------------------------
# Test cases: (filename, expected cyclomatic complexity)
# -------------------------------
test_cases = [
    ("cpp/OLD_simple.cpp", 4, 4),
    ("cpp/edge.cpp", 6, 6),
    ("cpp/complex.cpp", 52, 52),
    ("cpp/hyper_complex.cpp", 84, 83),
]

@pytest.mark.parametrize("filename,expected_basic,expected_full", test_cases)
def test_cyclomatic_files(filename, expected_basic, expected_full):
    """
    Test cyclomatic complexity computation for multiple C++ source files.

    This is a parametrized pytest function that validates both basic and 
    full (CFG-based) cyclomatic complexity calculations for a set of test files.

    Args:
        filename (str): Relative path to the C++ source file under test.
        expected_basic (int): Expected cyclomatic complexity from the regex-based method.
        expected_full (int): Expected cyclomatic complexity from the CFG-based method.

    Raises:
        AssertionError: If either the basic or full cyclomatic complexity does not match
                        the expected value.
    """
    code = load_code(filename, TEST_FILES_DIR)
    assert regex_compute_cyclomatic(code) == expected_basic
    assert cfg_compute_cyclomatic(code, filename) == expected_full