# -----------------------------------------------------------------------------
# Halstead Metrics Test Suite for C++/CUDA/Kokkos/OpenCL Source Files
# -----------------------------------------------------------------------------
# This pytest module verifies the correctness of Halstead complexity metrics
# computation across multiple languages and sample files. Metrics tested
# include:
# - Basic counts: n1, n2, N1, N2
# - Derived metrics: vocabulary, size, volume, difficulty, effort, time
#
# Includes:
# - Parametrized tests for multiple files and languages.
# - Rounds floating-point metrics to 2 decimals for assertions.
# - Provides detailed assertion messages for mismatches.
# -----------------------------------------------------------------------------
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
        "lang": "cpp",
        "expected": {
            "n1": 22, "n2": 15, "N1": 57, "N2": 30,
            "vocabulary": 37, "size": 87, "volume": 453.22,
            "difficulty": 22.00, "effort": 9970.89, "time": 553.94
        }
    },
    {
        "filename": "cpp/complex.cpp",
        "lang": "cpp",
        "expected": {
            "n1": 89, "n2": 99, "N1": 952, "N2": 469,
            "vocabulary": 188, "size": 1421, "volume": 10735.07,
            "difficulty": 210.81, "effort": 2263093.88, "time": 125727.44
        }
    },
    {
        "filename": "cuda/complex_cuda.cu",
        "lang": "cuda",
        "expected": {
            "n1": 38, "n2": 19, "N1": 280, "N2": 75,
            "vocabulary": 57, "size": 355, "volume": 2070.68,
            "difficulty": 75.00, "effort": 155300.70, "time": 8627.82
        }
    },
    {
        "filename": "kokkos/complex_kokkos.cpp",
        "lang": "kokkos",
        "expected": {
            "n1": 27, "n2": 25, "N1": 135, "N2": 63,
            "vocabulary": 52, "size": 198, "volume": 1128.69,
            "difficulty": 34.02, "effort": 38397.93, "time": 2133.22
        }
    },
    {
        "filename": "opencl/complex_opencl.cpp",
        "lang": "opencl",
        "expected": {
            "n1": 57, "n2": 48, "N1": 346, "N2": 147,
            "vocabulary": 105, "size": 493, "volume": 3310.12,
            "difficulty": 87.28, "effort": 288911.68, "time": 16050.65
        }
    },
]

@pytest.mark.parametrize("case", test_cases)
def test_halstead_metrics(case):
    """Tests Halstead metrics computation for multiple source files and languages.

    This function loads a source file, computes its Halstead metrics using
    `halstead_metrics`, and verifies both basic counts and derived metrics
    against expected values. 

    Args:
        case (dict): A dictionary containing:
            - "filename" (str): Path to the source file relative to the test samples directory.
            - "lang" (str): Programming language or framework (e.g., 'cpp', 'cuda').
            - "expected" (dict): Expected Halstead metrics with keys:
                'n1', 'n2', 'N1', 'N2', 'vocabulary', 'size', 'volume',
                'difficulty', 'effort', 'time'.

    Raises:
        AssertionError: If any computed metric does not match the expected value.
    """
    code = load_code(case["filename"], TEST_FILES_DIR)
    metrics = halstead_metrics(code, lang=case["lang"])
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