# -----------------------------------------------------------------------------
# Code Complexity Metrics Collection for Source Files
# -----------------------------------------------------------------------------
# This module provides functions to scan C++/CUDA source files or directories
# and compute a variety of complexity metrics per file.
#
# Includes:
# - Source Lines of Code (SLOC)
# - Nesting Depth
# - Cyclomatic Complexity (regex-based)
# - Cognitive Complexity (regex-based)
# - Halstead Metrics:
#   - Difficulty
#   - Volume
#   - Effort
# - GPU-native complexity deltas for detected frameworks (CUDA, OpenMP, etc.)
#
# Note:
# - Works with structured records represented as dictionaries.
# - Files with unsupported encodings are read with UTF-8, ignoring errors.
# - Designed for robustness: missing optional metrics do not break processing.
# -----------------------------------------------------------------------------
import os
from code_complexity.metrics.sloc import *
from code_complexity.metrics.nesting_depth import *
from code_complexity.metrics.cyclomatic import *
from code_complexity.metrics.cognitive import *
from code_complexity.metrics.halstead import *
from code_complexity.metrics.utils import detect_parallel_framework
from pathlib import Path


def collect_metrics(root_path: str):
    """
    Scan a file or directory and compute complexity metrics.

    Args:
        root_path (str): Path to a file or directory.

    Returns:
        list[dict]: List of metric dictionaries for each processed file.
    """
    records = []

    def analyze_file(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        sloc_val = compute_sloc(code)
        nesting = compute_nesting_depth(code)
        cyclomatic_val = regex_compute_cyclomatic(code)
        cognitive_val = regex_compute_cognitive(code)
        halstead_difficulty = difficulty(halstead_metrics(code, "auto"))
        halstead_volume = volume(halstead_metrics(code, "auto"))
        halstead_effort = effort(halstead_metrics(code, "auto"))
        
        # Base Halstead metrics (C++ reference)
        halstead_base = halstead_metrics(code, "cpp")
        halstead_difficulty_base = difficulty(halstead_base)
        halstead_volume_base = volume(halstead_base)
        
        # Detect languages/frameworks
        languages = detect_parallel_framework(code)
        # Compute GPU-native Halstead complexities
        gpu_complexity = {}
        for lang in languages:
            if lang != "cpp":
            #if lang in lang_to_fn and lang != 'cpp':  # skip base C++
                halstead_lang = halstead_metrics(code, lang)
                #halstead_lang = lang_to_fn[lang](code)
                gpu_complexity[lang] = {
                    "difficulty": difficulty(halstead_lang) - halstead_difficulty_base,
                    "volume": volume(halstead_lang) - halstead_volume_base,
                }
        
        return {
            "file": filepath,
            "sloc": sloc_val,
            "nesting": nesting,
            "cognitive": cognitive_val,
            "cyclomatic": cyclomatic_val,
            "halstead_difficulty": halstead_difficulty,
            "halstead_volume": halstead_volume,
            "halstead_effort": halstead_effort,
            "gpu_complexity": gpu_complexity,
        }
    # Handle single file
    if os.path.isfile(root_path) and root_path.endswith((".cpp", ".cu")):
        records.append(analyze_file(root_path))
    # Handle directory
    elif os.path.isdir(root_path):
        for subdir, _, files in os.walk(root_path):
            for file in files:
                if file.endswith((".cpp", ".cu")):
                    filepath = os.path.join(subdir, file)
                    records.append(analyze_file(filepath))
    return records

'''
    "file": "example.cu",
    "sloc": 120,
    "nesting": 3,
    "cognitive": 15,
    "cyclomatic": 8,
    "halstead_difficulty": 20,
    "halstead_volume": 1500,
    "gpu_complexity": {
        "cuda": {"difficulty": 10, "volume": 600},
        "openmp": {"difficulty": 5, "volume": 100}
    }
'''