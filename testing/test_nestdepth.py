import os
import pytest
from code_complexity.metrics.nesting_depth import compute_nesting_depth
from code_complexity.metrics.utils import load_code

# Directory containing test files for code complexity analysis - Using absolute path
TEST_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

# -------------------------------
# Test cases: (filename, expected nesting depth)
# -------------------------------
test_cases = [
    ("cpp/OLD_simple.cpp", 2),
    ("cpp/edge.cpp", 1),
    ("complex/complex.cpp", 4),
    ("complex/hyper_complex.cpp", 5),
]

@pytest.mark.parametrize("filename,expected_depth", test_cases)
def test_nesting_depth(filename, expected_depth):
    """Parametrized test for nesting depth across multiple C++ files."""
    code = load_code(filename, TEST_FILES_DIR)
    detected_depth = compute_nesting_depth(code)
    assert detected_depth == expected_depth, (
        f"File {filename}: detected {detected_depth}, expected {expected_depth}"
    )
