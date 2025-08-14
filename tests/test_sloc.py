import os
from projectFolder.metrics.sloc import compute_sloc

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")

def load_code(filename):
    with open(os.path.join(TEST_FILES_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()

def test_sloc_simple_cpp():
    code = load_code("OLD_simple.cpp")
    assert compute_sloc(code) == 15

def test_sloc_complex_cpp():
    code = load_code("complex.cpp")
    assert compute_sloc(code) == 39

#def test_sloc_basic():
#    assert compute_sloc("") == 0
#    assert compute_sloc("   ") == 0
#    assert compute_sloc("# This is a comment") == 0
#    assert compute_sloc("print('Hello')") == 1
#    assert compute_sloc("# Comment\nprint('Hello')") == 1
#    assert compute_sloc("print('Hello')\n# Another comment") == 1
#    assert compute_sloc("# Comment\n# Another comment") == 0
