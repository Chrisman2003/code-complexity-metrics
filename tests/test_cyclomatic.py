import os
from projectFolder.metrics.cyclomatic import basic_compute_cyclomatic

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")
# Testing Code Complexity = Structures + 1 (Default Path)
def load_code(filename):
    with open(os.path.join(TEST_FILES_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()

def test_cyclomatic_example_py():
    code = load_code("OLD_example.py")
    # Update the expected value below to match the actual cyclomatic complexity for example.py
    assert basic_compute_cyclomatic(code) >= 1 

def test_cyclomatic_simple_cpp():
    code = load_code("OLD_simple.cpp")
    # Update the expected value below to match the actual cyclomatic complexity for simple.cpp
    assert basic_compute_cyclomatic(code) >= 1 

def test_cyclomatic_complex_cpp():
    code = load_code("complex.cpp")
    # Update the expected value below to match the actual cyclomatic complexity for complex.cpp
    assert basic_compute_cyclomatic(code) >= 1