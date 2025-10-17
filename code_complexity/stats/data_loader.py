# Functions to load CSV/JSON or other data formats
import os
from code_complexity.metrics.sloc import *
from code_complexity.metrics.nesting_depth import *
from code_complexity.metrics.cyclomatic import *
from code_complexity.metrics.cognitive import *
from code_complexity.metrics.halstead import *
'''
def collect_metrics(root_dir: str):
    """Scan source files and compute complexity metrics."""
    records = []
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".cpp", ".cu")):
                filepath = os.path.join(subdir, file)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                sloc = compute_sloc(code)
                nesting = compute_nesting_depth(code)
                cyclomatic = basic_compute_cyclomatic(code)
                cognitive = regex_compute_cognitive(code)
                halstead = effort(halstead_metrics_merged(code))
                records.append({
                    "file": filepath,
                    "sloc": sloc,
                    "halstead": halstead,
                    "nesting": nesting,
                    "cognitive": cognitive,
                    "cyclomatic": cyclomatic,
                    "lines": code.count("\n") + 1
                })
    return records
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
        halstead_val = effort(halstead_metrics_auto(code))
        return {
            "file": filepath,
            "sloc": sloc_val,
            "halstead": halstead_val,
            "nesting": nesting,
            "cognitive": cognitive_val,
            "cyclomatic": cyclomatic_val,
            "lines": code.count("\n") + 1
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