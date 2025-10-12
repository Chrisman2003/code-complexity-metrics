import argparse
import os
import time
import logging
import inspect
import sys
from code_complexity.metrics import clang_parallel, cyclomatic, sloc, halstead, cognitive, nesting_depth

# -------------------------------
# Logging setup
# -------------------------------
metrics_logger = logging.getLogger("metrics")
metrics_handler = logging.StreamHandler(sys.stdout) 
metrics_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
))
metrics_logger.addHandler(metrics_handler)
metrics_logger.setLevel(logging.INFO)

# -------------------------------
# Timed execution wrapper
# -------------------------------
def timed(func, *args, **kwargs):
    """Run a function and measure its execution time.

    Args:
        func (callable): Function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        tuple: A tuple (result, elapsed_time) where `result` is the function output
               and `elapsed_time` is the runtime in seconds.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

# -------------------------------
# Analyze a single code file
# -------------------------------
def analyze_code(file_path: str, halstead_func, cyclomatic_func, cognitive_func, gpu_baseline_func=None):
    """Analyze complexity metrics of a single source code file.

    Args:
        file_path (str): Path to the source code file.
        halstead_func (callable): Function to compute Halstead metrics.
        gpu_baseline_func (callable, optional): Function to compute baseline GPU metrics. Defaults to None.

    Logs:
        INFO level metrics for SLOC, Nesting Depth, Cyclomatic complexity, Halstead metrics,
        Cognitive Complexity, and GPU deltas if applicable.
    """
    try:
        with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
            code = file.read()
    except Exception as e:
        metrics_logger.error("Failed to read file %s: %s", file_path, e)
        return
    cyc_name = cyclomatic_func.__name__ # accounting for different parametrization of the 2 functions
    cog_name = cognitive_func.__name__ # accounting for different parametrization of the 2 functions
    # Compute metrics with timing
    sloc_count, sloc_time = timed(sloc.compute_sloc, code)
    nesting_count, nesting_time = timed(nesting_depth.compute_nesting_depth, code)
    if cyc_name == "basic_compute_cyclomatic":
        cyclomatic_complexity, cyclomatic_time = timed(cyclomatic_func, code)
    else:
        cyclomatic_complexity, cyclomatic_time = timed(cyclomatic_func, code, file_path)
    if cog_name == "regex_compute_cognitive":
        cognitive_complexity, cognitive_time = timed(cognitive_func, code) 
    else:
        cognitive_complexity, cognitive_time = timed(cognitive_func, file_path)
    halstead_metrics, halstead_time = timed(halstead_func, code)

    # Log results
    metrics_logger.info("Analyzing file: %s", file_path)
    metrics_logger.info("SLOC: %d  [runtime: %.4fs]", sloc_count, sloc_time)
    metrics_logger.info("Nesting Depth: %d  [runtime: %.4fs]", nesting_count, nesting_time)
    metrics_logger.info("Cyclomatic Complexity: %d  [runtime: %.4fs]", cyclomatic_complexity, cyclomatic_time)
    metrics_logger.info("Cognitive Complexity: %d  [runtime: %.4fs]", cognitive_complexity, cognitive_time) 
    metrics_logger.info("Halstead Metrics:")
    for k, v in halstead_metrics.items():
        metrics_logger.info("  %s: %s", k, v)
    metrics_logger.info("  Vocabulary: %s", halstead.vocabulary(halstead_metrics))
    metrics_logger.info("  Size: %s", halstead.size(halstead_metrics))
    metrics_logger.info("  Volume: %.2f", halstead.volume(halstead_metrics))
    metrics_logger.info("  Difficulty: %.2f", halstead.difficulty(halstead_metrics))
    metrics_logger.info("  Effort: %.2f", halstead.effort(halstead_metrics))
    metrics_logger.info("  Time: %.2fs", halstead.time(halstead_metrics))
    metrics_logger.info("[Halstead runtime: %.4fs]", halstead_time)

    # Compute GPU delta metrics if requested
    if gpu_baseline_func:
        baseline_metrics, _ = timed(gpu_baseline_func, code)
        delta = {k: halstead_metrics[k] - baseline_metrics[k] for k in halstead_metrics}
        metrics_logger.info("GPU Delta Metrics (GPU - C++ baseline):")
        for k, v in delta.items():
            metrics_logger.info("  %s: %s", k, v)
        metrics_logger.info("  Vocabulary delta: %s", halstead.vocabulary(halstead_metrics) - halstead.vocabulary(baseline_metrics))
        metrics_logger.info("  Size delta: %s", halstead.size(halstead_metrics) - halstead.size(baseline_metrics))
        metrics_logger.info("  Volume delta: %.2f", halstead.volume(halstead_metrics) - halstead.volume(baseline_metrics))
        metrics_logger.info("  Difficulty delta: %.2f", halstead.difficulty(halstead_metrics) - halstead.difficulty(baseline_metrics))
        metrics_logger.info("  Effort delta: %.2f", halstead.effort(halstead_metrics) - halstead.effort(baseline_metrics))
        metrics_logger.info("  Time delta: %.2fs", halstead.time(halstead_metrics) - halstead.time(baseline_metrics))

    metrics_logger.info("-" * 40)

# -------------------------------
# Analyze all files in a directory
# -------------------------------
def analyze_directory(directory_path: str, halstead_func, cyclomatic_func, cognitive_func,gpu_baseline_func=None):
    """Recursively analyze all source code files in a directory.

    Args:
        directory_path (str): Path to the directory.
        halstead_func (callable): Function to compute Halstead metrics.
        gpu_baseline_func (callable, optional): Function to compute baseline GPU metrics. Defaults to None.
    """
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith((".cpp", ".cxx", ".cc", ".cu", ".cl", ".hpp", ".h")):
                file_path = os.path.join(root, file_name)
                analyze_code(file_path, halstead_func, cyclomatic_func, cognitive_func, gpu_baseline_func=gpu_baseline_func)

# -------------------------------
# Main CLI
# -------------------------------
def main():
    """Command-line interface for analyzing code complexity metrics.

    Accepts a file or directory path and optional arguments:
        --lang: Language for Halstead metrics
        --gpu-delta: Compare GPU constructs vs C++ baseline
        --verbose: Enable debug logging
    """
    parser = argparse.ArgumentParser(description="Analyze code complexity metrics for a file or directory.")
    parser.add_argument("path", help="Path to the code file or directory to analyze")
    parser.add_argument("--lang", choices=[
        "cpp",
        "cuda",
        "opencl",
        "kokkos",
        "openmp",
        "adaptivecpp",
        "openacc",
        "opengl_vulkan",
        "webgpu",
        "boost",
        "metal",
        "merged"
    ], default="cpp",
                        help="Language extension for Halstead metrics")
    parser.add_argument("--gpu-delta", action="store_true",
                        help="Compute added complexity of GPU constructs vs C++ baseline")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--cyclomatic", choices=["advanced"], default="basic",
                        help="Cyclomatic complexity calculation method")
    parser.add_argument("--cognitive", choices=["advanced"], default="regex",
                        help="Cognitive complexity calculation method")
    args = parser.parse_args()
    # Enable debug logging if verbose
    if args.verbose:
        metrics_logger.setLevel(logging.DEBUG)

    # Validate path
    if not os.path.exists(args.path):
        metrics_logger.error("Path does not exist: %s", args.path)
        return

    # Select Halstead function based on language
    halstead_func = {
        "cpp": halstead.halstead_metrics_cpp,
        "cuda": halstead.halstead_metrics_cuda,
        "opencl": halstead.halstead_metrics_opencl,
        "kokkos": halstead.halstead_metrics_kokkos,
        "openmp": halstead.halstead_metrics_openmp,
        "adaptivecpp": halstead.halstead_metrics_adaptivecpp,
        "openacc": halstead.halstead_metrics_openacc,
        "opengl_vulkan": halstead.halstead_metrics_opengl_vulkan,
        "webgpu": halstead.halstead_metrics_webgpu,
        "boost": halstead.halstead_metrics_boost,
        "metal": halstead.halstead_metrics_metal,
        "merged": halstead.halstead_metrics_merged
    }[args.lang]
    # Select Cyclomatic function based on method
    cyclomatic_func = {
        "advanced": cyclomatic.compute_cyclomatic,
        "basic": cyclomatic.basic_compute_cyclomatic
    }[args.cyclomatic]
    # Select Cognitive function based on method
    cognitive_func = {
        "advanced": clang_parallel.compute_cognitive_complexity_file,
        "regex": cognitive.regex_compute_cognitive
    }[args.cognitive]
        
    # Determine baseline for GPU delta
    gpu_baseline_func = halstead.halstead_metrics_cpp if (args.gpu_delta and args.lang != "cpp") else None
    # Analyze file or directory
    if os.path.isfile(args.path):
        analyze_code(args.path, halstead_func, cyclomatic_func, cognitive_func, gpu_baseline_func=gpu_baseline_func)
    elif os.path.isdir(args.path):
        analyze_directory(args.path, halstead_func, cyclomatic_func, cognitive_func, gpu_baseline_func=gpu_baseline_func)
    else:
        metrics_logger.error("Path is neither a file nor a directory: %s", args.path)

if __name__ == "__main__":
    main()
