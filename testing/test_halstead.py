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
        "filename": "cuda/complex.cu",
        "lang": "cuda",
        "expected": {
            "n1": 38, "n2": 19, "N1": 280, "N2": 75,
            "vocabulary": 57, "size": 355, "volume": 2070.68,
            "difficulty": 75.00, "effort": 155300.70, "time": 8627.82
        }
    },
    {
        "filename": "kokkos/complex.cpp",
        "lang": "kokkos",
        "expected": {
            "n1": 27, "n2": 25, "N1": 135, "N2": 63,
            "vocabulary": 52, "size": 198, "volume": 1128.69,
            "difficulty": 34.02, "effort": 38397.93, "time": 2133.22
        }
    },
    {
        "filename": "opencl/complex.cpp",
        "lang": "opencl",
        "expected": {
            "n1": 57, "n2": 48, "N1": 346, "N2": 147,
            "vocabulary": 105, "size": 493, "volume": 3310.12,
            "difficulty": 87.28, "effort": 288911.68, "time": 16050.65
        }
    },
        {
        "filename": "adaptive_cpp/complex.cpp",
        "lang": "adaptivecpp",
        "expected": {
            "n1": 119, "n2": 185, "N1": 1747, "N2": 915,
            "vocabulary": 304, "size": 2662, "volume": 21955.98,
            "difficulty": 294.28, "effort": 6461289.77, "time": 358960.54
        }
    },
    # Boost
    {
        "filename": "boost/complex.cpp",
        "lang": "boost",
        "expected": {
            "n1": 86, "n2": 133, "N1": 940, "N2": 566,
            "vocabulary": 219, "size": 1506, "volume": 11708.83,
            "difficulty": 182.99, "effort": 2142627.73, "time": 119034.87
        }
    },
    # Metal
    {
        "filename": "metal/complex.cpp",
        "lang": "metal",
        "expected": {
            "n1": 70, "n2": 68, "N1": 426, "N2": 194,
            "vocabulary": 138, "size": 620, "volume": 4407.29,
            "difficulty": 99.85, "effort": 440080.39, "time": 24448.91
        }
    },
    # OpenACC
    {
        "filename": "openacc/complex.cpp",
        "lang": "openacc",
        "expected": {
            "n1": 79, "n2": 93, "N1": 789, "N2": 445,
            "vocabulary": 172, "size": 1234, "volume": 9164.01,
            "difficulty": 189.01, "effort": 1732047.29, "time": 96224.85
        }
    },
    # OpenGL/Vulkan
    {
        "filename": "opengl_vulkan/complex.cpp",
        "lang": "opengl_vulkan",
        "expected": {
            "n1": 162, "n2": 159, "N1": 930, "N2": 409,
            "vocabulary": 321, "size": 1339, "volume": 11149.09,
            "difficulty": 208.36, "effort": 2323007.37, "time": 129055.97
        }
    },
    # OpenMP
    {
        "filename": "openmp/complex.cpp",
        "lang": "openmp",
        "expected": {
            "n1": 84, "n2": 91, "N1": 762, "N2": 401,
            "vocabulary": 175, "size": 1163, "volume": 8665.76,
            "difficulty": 185.08, "effort": 1603831.92, "time": 89101.77
        }
    },
    # Thrust
    {
        "filename": "thrust/complex.cpp",
        "lang": "thrust",
        "expected": {
            "n1": 52, "n2": 52, "N1": 426, "N2": 181,
            "vocabulary": 104, "size": 607, "volume": 4067.17,
            "difficulty": 90.50, "effort": 368078.61, "time": 20448.81
        }
    },
    # WebGPU
    {
        "filename": "webgpu/complex.cpp",
        "lang": "webgpu",
        "expected": {
            "n1": 91, "n2": 117, "N1": 840, "N2": 398,
            "vocabulary": 208, "size": 1238, "volume": 9533.14,
            "difficulty": 154.78, "effort": 1475518.90, "time": 81973.27
        }
    }
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
