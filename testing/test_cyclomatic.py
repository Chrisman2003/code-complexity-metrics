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

def test_cyclomatic_simple_cpp():
    """Tests cyclomatic complexity calculation on a simple C++ file.

    The expected complexity is at least 1.
    """
    code = load_code("old/OLD_simple.cpp")
    assert basic_compute_cyclomatic(code) == 3


def test_cyclomatic_complex_cpp():
    """Tests cyclomatic complexity calculation on a more complex C++ file.

    The expected complexity is at least 1.
    """
    code = load_code("complex/complex.cpp")
    assert basic_compute_cyclomatic(code) >= 1

'''
Edge Cases for Commenting:
1) 
std::string s = "This is not a // comment";
char c = '/';
std::string t = "/* not a comment */";
2)
/* Outer comment
   /* Inner comment */
   End of outer */
3) 
#define STR(x) "/* " #x " */"
'''

'''
Edge Cases: 
-> label: 
-> if (x > 0) 
-> } if (x > 0)
'''