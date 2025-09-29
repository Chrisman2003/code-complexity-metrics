import os
from code_complexity.metrics.cyclomatic import *

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
    
def test_keep_kernel_strings_and_code_opencl():
    code = load_code("opencl_edge_cases.cpp")
    cleaned = remove_string_literals(code)    
    # assert preserved kernel
    assert "__kernel void add" in cleaned
    # assert raw kernel kept
    assert "__kernel void multiply" in cleaned
    # assert random string removed
    assert "This is just a string with no kernel" not in cleaned
    # assert character literal removed
    assert "'x'" not in cleaned
    # assert tricky escape kept
    assert "trickyEscapes" in cleaned and "__kernel" in cleaned


'''
Potential Edge Case: string prefixed to __kernel in a wrapped string
'''