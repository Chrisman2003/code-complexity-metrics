import os
import pytest
from code_complexity.metrics.cyclomatic import remove_string_literals
from code_complexity.metrics.utils import load_code

# Directory containing test files for code complexity analysis
TEST_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

# -------------------------------
# Test cases: (filename)
# -------------------------------
test_files = [
    "opencl/opencl_edge_cases.cpp",
    # Add more edge-case files here if needed
]

@pytest.mark.parametrize("filename", test_files)
def test_tools(filename):
    """Ensure kernel code is preserved while unrelated strings are removed."""
    code = load_code(filename, TEST_FILES_DIR)
    cleaned = remove_string_literals(code)

    # Assertions
    # Preserve kernel functions
    assert "__kernel void add" in cleaned
    assert "__kernel void multiply" in cleaned

    # Remove unrelated strings
    assert "This is just a string with no kernel" not in cleaned
    assert "'x'" not in cleaned

    # Preserve tricky escapes and kernel keyword
    assert "trickyEscapes" in cleaned and "__kernel" in cleaned
