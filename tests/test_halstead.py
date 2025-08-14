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
    assert metrics['n1'] == 19
    assert metrics['n2'] == 19
    assert metrics['N1'] == 47
    assert metrics['N2'] == 34
    assert vocabulary(metrics) == 38
    assert size(metrics) == 81
    assert round(volume(metrics),2) ==425.08
    assert round(difficulty(metrics),2) == 17.00
    assert round(effort(metrics),2) == 7226.40
    assert round(time(metrics),2) == 401.47
    
def test_halstead_complex_cpp():
    code = load_code("complex.cpp")
    metrics = halstead_metrics_cpp(code)
    # Update the expected values below to match the actual Halstead metrics for complex.cpp
    assert metrics['n1'] == 39
    assert metrics['n2'] == 96
    assert metrics['N1'] == 227
    assert metrics['N2'] == 241
    assert vocabulary(metrics) == 135
    assert size(metrics) == 468
    assert round(volume(metrics),2) == 3311.95
    assert round(difficulty(metrics),2) == 48.95
    assert round(effort(metrics),2) == 162130.29
    assert round(time(metrics),2) == 9007.24
    
def test_halstead_complex_cuda():
    code = load_code("complex_cuda.cu")
    metrics = halstead_metrics_cuda(code)
    # Update the expected values below to match the actual Halstead metrics for complex.cpp
    assert metrics['n1'] == 34
    assert metrics['n2'] == 38
    assert metrics['N1'] == 258
    assert metrics['N2'] == 113
    assert vocabulary(metrics) == 72
    assert size(metrics) == 371
    assert round(volume(metrics),2) == 2289.04
    assert round(difficulty(metrics),2) == 50.55
    assert round(effort(metrics),2) == 115717.11
    assert round(time(metrics),2) == 6428.73

def test_halstead_complex_kokkos():
    code = load_code("complex_kokkos.cpp")
    metrics = halstead_metrics_kokkos(code)
    # Update the expected values below to match the actual Halstead metrics for complex.cpp
    assert metrics['n1'] == 26
    assert metrics['n2'] == 46
    assert metrics['N1'] == 153
    assert metrics['N2'] == 90
    assert vocabulary(metrics) ==72
    assert size(metrics) == 243
    assert round(volume(metrics),2) == 1499.29
    assert round(difficulty(metrics),2) == 25.43
    assert round(effort(metrics),2) == 38134.16
    assert round(time(metrics),2) == 2118.56

def test_halstead_complex_opencl():
    code = load_code("complex_opencl.cpp")
    metrics = halstead_metrics_opencl(code)
    # Update the expected values below to match the actual Halstead metrics for complex.cpp
    assert metrics['n1'] == 39
    assert metrics['n2'] == 119
    assert metrics['N1'] == 333
    assert metrics['N2'] == 280
    assert vocabulary(metrics) == 158
    assert size(metrics) == 613
    assert round(volume(metrics),2) == 4477.22
    assert round(difficulty(metrics),2) == 45.88
    assert round(effort(metrics),2) == 205425.28
    assert round(time(metrics),2) == 11412.52
