import os
import pytest
from code_complexity.metrics.halstead import *
from code_complexity.metrics.utils import load_code

# Directory containing test files for code complexity analysis - Using absolute path
TEST_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

# -------------------------------
# Test cases: each entry includes filename, function, and expected metrics
# -------------------------------
test_cases = [
    {
        "filename": "cpp/OLD_simple.cpp",
        "func": halstead_metrics_cpp,
        "expected": {
            "n1": 23, "n2": 13, "N1": 58, "N2": 29,
            "vocabulary": 36, "size": 87, "volume": 449.78,
            "difficulty": 25.65, "effort": 11538.68, "time": 641.04
        }
    },
    {
        "filename": "complex/complex.cpp",
        "func": halstead_metrics_cpp,
        "expected": {
            "n1": 60, "n2": 87, "N1": 885, "N2": 495,
            "vocabulary": 147, "size": 1380, "volume": 9935.55,
            "difficulty": 170.69, "effort": 1695895.23, "time": 94216.40
        }
    },
    {
        "filename": "complex/complex_cuda.cu",
        "func": halstead_metrics_cuda,
        "expected": {
            "n1": 39, "n2": 18, "N1": 275, "N2": 75,
            "vocabulary": 57, "size": 350, "volume": 2041.51,
            "difficulty": 81.25, "effort": 165872.81, "time": 9215.16
        }
    },
    {
        "filename": "complex/complex_kokkos.cpp",
        "func": halstead_metrics_kokkos,
        "expected": {
            "n1": 25, "n2": 18, "N1": 127, "N2": 62,
            "vocabulary": 43, "size": 189, "volume": 1025.56,
            "difficulty": 43.06, "effort": 44156.23, "time": 2453.12
        }
    },
    {
        "filename": "complex/complex_opencl.cpp",
        "func": halstead_metrics_opencl,
        "expected": {
            "n1": 53, "n2": 46, "N1": 329, "N2": 146,
            "vocabulary": 99, "size": 475, "volume": 3148.94,
            "difficulty": 84.11, "effort": 264853.61, "time": 14714.09
        }
    },
]

@pytest.mark.parametrize("case", test_cases)
def test_halstead_metrics(case):
    """Parametrized test for Halstead metrics across languages and files."""
    code = load_code(case["filename"], TEST_FILES_DIR)
    metrics = case["func"](code)
    expected = case["expected"]
    # Assertions for basic counts
    assert metrics['n1'] == expected['n1'], f"{case['filename']} n1 mismatch"
    assert metrics['n2'] == expected['n2'], f"{case['filename']} n2 mismatch"
    assert metrics['N1'] == expected['N1'], f"{case['filename']} N1 mismatch"
    assert metrics['N2'] == expected['N2'], f"{case['filename']} N2 mismatch"

    # Assertions for derived metrics (rounding floats to 2 decimals)
    assert vocabulary(metrics) == expected['vocabulary'], f"{case['filename']} vocabulary mismatch"
    assert size(metrics) == expected['size'], f"{case['filename']} size mismatch"
    assert round(volume(metrics), 2) == expected['volume'], f"{case['filename']} volume mismatch"
    assert round(difficulty(metrics), 2) == expected['difficulty'], f"{case['filename']} difficulty mismatch"
    assert round(effort(metrics), 2) == expected['effort'], f"{case['filename']} effort mismatch"
    assert round(time(metrics), 2) == expected['time'], f"{case['filename']} time mismatch"
 
'''
Edge Case:
Boundary Word Wrapping for keywords searched by the Halstead Function ?
'''