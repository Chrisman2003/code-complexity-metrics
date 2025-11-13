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
    ("complex/complex.cpp", 38),
    ("complex/hyper_complex.cpp", 90),
]

@pytest.mark.parametrize("filename,expected", test_cases)
def test_cognitive_complexity(filename, expected):
    """Parametrized test for cognitive complexity."""
    code = load_code(filename, TEST_FILES_DIR)
    detected = regex_compute_cognitive(code)
    assert detected == expected, f"File {filename}: detected {detected}, expected {expected}"
