import os
import pytest
from code_complexity.metrics.sloc import compute_sloc
from code_complexity.metrics.utils import load_code

# Directory containing test files for SLOC calculation
TEST_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

# -------------------------------
# Test cases: (filename, expected SLOC)
# -------------------------------
test_cases = [
    ("cpp/OLD_simple.cpp", 17),
    ("cpp/edge.cpp", 11),
    ("cpp/complex.cpp", 158),
    ("cpp/hyper_complex.cpp", 209),
]

@pytest.mark.parametrize("filename,expected_sloc", test_cases)
def test_sloc(filename, expected_sloc):
    """Parametrized test for SLOC across multiple C++ files."""
    code = load_code(filename, TEST_FILES_DIR)
    detected_sloc = compute_sloc(code)
    assert detected_sloc == expected_sloc, (
        f"File {filename}: detected {detected_sloc}, expected {expected_sloc}"
    )
