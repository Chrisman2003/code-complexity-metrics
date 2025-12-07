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
    "opencl/edge_cases.cpp",
    # Add more edge-case files here if needed
]

@pytest.mark.parametrize("filename", test_files)
def test_tools(filename):
    """Tests removal of unrelated string literals while preserving kernel code.

    This function loads a source file, applies `remove_string_literals` to
    remove non-code string literals, and asserts that:

    - Kernel functions (e.g., "__kernel void add") are preserved.
    - Unrelated strings are removed.
    - Tricky escape sequences and relevant keywords are retained.

    Args:
        filename (str): Path to the source file relative to the test samples directory.

    Raises:
        AssertionError: If kernel functions are removed, unrelated strings remain,
                        or expected tricky sequences are not preserved.
    """
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
