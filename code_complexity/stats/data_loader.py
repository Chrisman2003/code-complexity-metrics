import os
from code_complexity.metrics.sloc import *
from code_complexity.metrics.nesting_depth import *
from code_complexity.metrics.cyclomatic import *
from code_complexity.metrics.cognitive import *
from code_complexity.metrics.halstead import *
from collections import Counter
from pathlib import Path

'''
✅ Ranked from most to least accurate for total file complexity:
1) E — Effort
2) D — Difficulty
3) V — Volume
'''
def collect_metrics(root_path: str):
    """Scan a file or directory and compute complexity metrics."""
    records = []

    def analyze_file(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        sloc_val = compute_sloc(code)
        nesting = compute_nesting_depth(code)
        cyclomatic_val = basic_compute_cyclomatic(code)
        cognitive_val = regex_compute_cognitive(code)
        halstead_difficulty = difficulty(halstead_metrics_auto(code, Path(filepath).suffix))
        halstead_volume = volume(halstead_metrics_auto(code, Path(filepath).suffix))
        
        # Base Halstead metrics (C++ reference)
        halstead_base = halstead_metrics_cpp(code, Path(filepath).suffix)
        halstead_difficulty_base = difficulty(halstead_base)
        halstead_volume_base = volume(halstead_base)
        
        # Detect languages/frameworks
        languages = detect_parallel_framework(code, Path(filepath).suffix)
        # Mapping of language/framework to metric function
        lang_to_fn = {
            'cpp': halstead_metrics_cpp,
            'cuda': halstead_metrics_cuda,
            'kokkos': halstead_metrics_kokkos,
            'opencl': halstead_metrics_opencl,
            'openmp': halstead_metrics_openmp,
            'adaptivecpp': halstead_metrics_adaptivecpp,
            'openacc': halstead_metrics_openacc,
            'opengl_vulkan': halstead_metrics_opengl_vulkan,
            'webgpu': halstead_metrics_webgpu,
            'boost': halstead_metrics_boost,
            'metal': halstead_metrics_metal,
            'thrust': halstead_metrics_thrust,
            'slang': halstead_metrics_slang,
        }

        # Compute GPU-native Halstead complexities
        gpu_complexity = {}
        for lang in languages:
            if lang in lang_to_fn and lang != 'cpp':  # skip base C++
                halstead_lang = lang_to_fn[lang](code)
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
            # Halstead Metric
            "halstead_difficulty": halstead_difficulty,
            "halstead_volume": halstead_volume,
            "gpu_complexity": gpu_complexity,
        }
    # Handle single file
    if os.path.isfile(root_path) and root_path.endswith((".cpp", ".cu", ".slang")):
        records.append(analyze_file(root_path))
    # Handle directory
    elif os.path.isdir(root_path):
        for subdir, _, files in os.walk(root_path):
            for file in files:
                if file.endswith((".cpp", ".cu", ".slang")):
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