# Functions to load CSV/JSON or other data formats
import os
from code_complexity.metrics.sloc import *
from code_complexity.metrics.nesting_depth import *
from code_complexity.metrics.cyclomatic import *
from code_complexity.metrics.cognitive import *
from code_complexity.metrics.halstead import *

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
