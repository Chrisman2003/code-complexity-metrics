import argparse
import os
import time
import logging
from code_complexity.metrics import sloc, halstead, cyclomatic

# -------------------------------
# Logging setup
# -------------------------------
logger = logging.getLogger("code_complexity")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -------------------------------
# Timed execution wrapper
# -------------------------------
def timed(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

# -------------------------------
# Analyze a single code file
# -------------------------------
def analyze_code(file_path: str, halstead_func, gpu_baseline_func=None):
    try:
        with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
            code = file.read()
    except Exception as e:
        logger.error("Failed to read file %s: %s", file_path, e)
        return

    # Compute metrics with timings
    sloc_count, sloc_time = timed(sloc.compute_sloc, code)
    halstead_metrics, halstead_time = timed(halstead_func, code)
    cyclomatic_complexity, cyclomatic_time = timed(cyclomatic.compute_cyclomatic, code)

    logger.info("Analyzing file: %s", file_path)
    logger.info("SLOC: %d  [runtime: %.4fs]", sloc_count, sloc_time)
    logger.info("Cyclomatic Complexity: %d  [runtime: %.4fs]", cyclomatic_complexity, cyclomatic_time)

    logger.info("Halstead Metrics:")
    for k, v in halstead_metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("  Vocabulary: %s", halstead.vocabulary(halstead_metrics))
    logger.info("  Size: %s", halstead.size(halstead_metrics))
    logger.info("  Volume: %.2f", halstead.volume(halstead_metrics))
    logger.info("  Difficulty: %.2f", halstead.difficulty(halstead_metrics))
    logger.info("  Effort: %.2f", halstead.effort(halstead_metrics))
    logger.info("  Time: %.2fs", halstead.time(halstead_metrics))
    logger.info("[Halstead runtime: %.4fs]", halstead_time)

    # Compute GPU delta if requested
    if gpu_baseline_func:
        baseline_metrics, _ = timed(gpu_baseline_func, code)
        delta = {k: halstead_metrics[k] - baseline_metrics[k] for k in halstead_metrics}
        logger.info("GPU Delta Metrics (GPU - C++ baseline):")
        for k, v in delta.items():
            logger.info("  %s: %s", k, v)
        logger.info("  Vocabulary delta: %s", halstead.vocabulary(halstead_metrics) - halstead.vocabulary(baseline_metrics))
        logger.info("  Size delta: %s", halstead.size(halstead_metrics) - halstead.size(baseline_metrics))
        logger.info("  Volume delta: %.2f", halstead.volume(halstead_metrics) - halstead.volume(baseline_metrics))
        logger.info("  Difficulty delta: %.2f", halstead.difficulty(halstead_metrics) - halstead.difficulty(baseline_metrics))
        logger.info("  Effort delta: %.2f", halstead.effort(halstead_metrics) - halstead.effort(baseline_metrics))
        logger.info("  Time delta: %.2fs", halstead.time(halstead_metrics) - halstead.time(baseline_metrics))

    logger.info("-" * 40)

# -------------------------------
# Analyze all files in a directory
# -------------------------------
def analyze_directory(directory_path: str, halstead_func, gpu_baseline_func=None):
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith((".cpp", ".cxx", ".cc", ".cu", ".cl", ".hpp", ".h")):
                file_path = os.path.join(root, file_name)
                analyze_code(file_path, halstead_func, gpu_baseline_func=gpu_baseline_func)

# -------------------------------
# Main CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze code complexity metrics for a file or directory.")
    parser.add_argument("path", help="Path to the code file or directory to analyze")
    parser.add_argument("--lang", choices=["cuda", "opencl", "kokkos", "cpp", "merged"], default="cpp",
                        help="Language extension for Halstead metrics")
    parser.add_argument("--gpu-delta", action="store_true",
                        help="Compute added complexity of GPU constructs vs C++ baseline")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not os.path.exists(args.path):
        logger.error("Path does not exist: %s", args.path)
        return

    # Select Halstead function based on language
    halstead_func = {
        "cuda": halstead.halstead_metrics_cuda,
        "opencl": halstead.halstead_metrics_opencl,
        "kokkos": halstead.halstead_metrics_kokkos,
        "merged": halstead.halstead_metrics_merged,
        "cpp": halstead.halstead_metrics_cpp
    }[args.lang]

    # Determine baseline for GPU delta
    gpu_baseline_func = halstead.halstead_metrics_cpp if (args.gpu_delta and args.lang != "cpp") else None

    # Analyze
    if os.path.isfile(args.path):
        analyze_code(args.path, halstead_func, gpu_baseline_func=gpu_baseline_func)
    elif os.path.isdir(args.path):
        analyze_directory(args.path, halstead_func, gpu_baseline_func=gpu_baseline_func)
    else:
        logger.error("Path is neither a file nor a directory: %s", args.path)

if __name__ == "__main__":
    main()
