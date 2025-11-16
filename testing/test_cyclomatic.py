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
    ("complex/complex.cpp", 52, 52),
    ("complex/hyper_complex.cpp", 83, 83),
]

@pytest.mark.parametrize("filename,expected_basic,expected_full", test_cases)
def test_cyclomatic_files(filename, expected_basic, expected_full):
    code = load_code(filename, TEST_FILES_DIR)
    assert regex_compute_cyclomatic(code) == expected_basic
    assert cfg_compute_cyclomatic(code, filename) == expected_full