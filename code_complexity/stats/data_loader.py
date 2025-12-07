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

'''
EDGE CASE DOCUMENTATION:
IMPORTANT:
1) The Halstead Delta Computation is only designed for files containing a singular GPU framework.
    In files where multiple GPU frameworks are used, the delta values for Halstead metrics
    may have discrepancies when reflecting the individual contributions of each framework 
    (through the operand set). The C++ lens inherently subtracts from operand set all framework
    functions (cudaMalloc() and thrust::...). A thrust lens will not subtract cudaMalloc() calls from the operand
    set for example. 
    -> This is an inevitable edge case. Since one can't subtract a merged set with all framework functions from
    the operand set. Any attempt to merge framework vocabularies for subtraction would distort the lens-specific 
    meaning of the metrics.
    -> Mixed-framework files often serve as glue code, integration layers, or migration scaffolding rather than 
    representative algorithmic implementations. Their Halstead deltas are not expected to reflect clean 
    framework-specific complexity because the file's purpose itself is not entirely framework-specific.
    -> For manual framework complexity deltas, one can modify the detection framework by changing the corresponding
    library headers.
2) Files that do not include any GPU framework will not have any entries in the "gpu_complexity" dictionary.
'''