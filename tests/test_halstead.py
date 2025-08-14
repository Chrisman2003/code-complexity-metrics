import os
from projectFolder.metrics.halstead import*

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")

def load_code(filename):
    with open(os.path.join(TEST_FILES_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()

def test_halstead_simple_cpp():
    code = load_code("OLD_simple.cpp")
    metrics = halstead_metrics_cpp(code)
    # Update the expected values below to match the actual Halstead metrics for simple.cpp
    assert metrics['n1'] == 14
    assert metrics['n2'] == 18
    assert metrics['N1'] == 37
    assert metrics['N2'] == 33
    assert vocabulary(metrics) == 32
    assert size(metrics) == 70
    assert round(volume(metrics),2) == 350.00
    assert round(difficulty(metrics),2) == 12.83
    assert round(effort(metrics),2) == 4491.67
    assert round(time(metrics),2) == 249.54
    
def test_halstead_complex_cpp():
    code = load_code("complex.cpp")
    metrics = halstead_metrics_cpp(code)
    # Update the expected values below to match the actual Halstead metrics for complex.cpp
    assert metrics['n1'] == 30
    assert metrics['n2'] == 54
    assert metrics['N1'] == 152
    assert metrics['N2'] == 117
    
    assert vocabulary(metrics) == 84
    assert size(metrics) == 269
    assert round(volume(metrics),2) == 1719.53
    assert round(difficulty(metrics),2) == 32.50
    assert round(effort(metrics),2) == 55884.84
    assert round(time(metrics),2) == 3104.71
    
