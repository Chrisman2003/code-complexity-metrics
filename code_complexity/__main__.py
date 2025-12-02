# -----------------------------------------------------------------------------
# Code Complexity Analysis CLI
# -----------------------------------------------------------------------------
# This module provides a command-line interface for analyzing source code
# complexity metrics across multiple languages and frameworks.
# Includes:
# - Computing multiple code complexity metrics per file:
#   - Source Lines of Code (SLOC)
#   - Nesting Depth
#   - Cyclomatic Complexity (regex or CFG-based)
#   - Cognitive Complexity
#   - Halstead Metrics (with optional GPU delta comparison)
# - Recursively analyzes directories or single files.
# - Generates statistical reports (basic or advanced) as PDF.
# - Timed execution for each metric to profile performance.
# Usage:
#   code-metrics <path> [--lang LANG] [--report basic|advanced] [--gpu-delta] [-v]
# -----------------------------------------------------------------------------
import argparse
import os
import time
from code_complexity.metrics import cyclomatic, sloc, halstead, cognitive, nesting_depth
from code_complexity.metrics.utils import metrics_logger, plain_logger
from code_complexity.stats.data_loader import collect_metrics
from code_complexity.stats.analysis import summarize
from code_complexity.stats.report_generator import *
import logging

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
    return result, end-start

def stat_report(root_dir: str, report_func):
    """Generates a statistical report of code complexity metrics for a given directory.

    This function collects code complexity metrics from the specified directory,
    summarizes them, computes correlations, and generates a report as a PDF file.
    It supports either a basic or advanced reporting pipeline, depending on the
    provided `report_func`.

    Args:
        root_dir (str): Path to the root directory containing source code files to analyze.
        report_func (callable): Function to generate the report. Should be either
            `generate_basic_report` or `generate_advanced_report`.

    Logs:
        Info-level messages are logged at each major step of the pipeline, including:
        - Start of the analysis.
        - Number of records collected.
        - Summaries and correlations.
        - Report generation and completion.

    Output:
        A PDF report named "complexity_report.pdf" is generated in the current working directory.
    """
    if report_func == generate_basic_report:
        metrics_logger.info("📊 Running basic statistical analysis pipeline...")
    else:
        metrics_logger.info("📊 Running advanced statistical analysis pipeline...")
    
    metrics_logger.info("🔍 Collecting metrics from: %s", root_dir)
    records = collect_metrics(root_dir)
    metrics_logger.info("✅ Collected %d records\n", len(records))

    metrics_logger.info("📊 Summarizing metrics...")
    summary, correlations = summarize(records)
    metrics_logger.info("\nSummary:\n%s", summary)
    metrics_logger.info("\nCorrelations:\n%s", correlations)

    metrics_logger.info("📈 Generating report...")
    if report_func == generate_basic_report:
        generate_basic_report(records, output_path="complexity_report.pdf")
    else:
        generate_advanced_report(records, output_path="complexity_report.pdf")
    metrics_logger.info("✅ Report saved as complexity_report.pdf")

def analyze_code(file_path: str, halstead_lang, cyclomatic_func, gpu_delta_enabled):
    """Analyze complexity metrics of a single source code file.

    Args:
        file_path (str): Path to the source code file.
        halstead_func (callable): Function to compute Halstead metrics.
        gpu_delta_enabled (boolean): Flag whether to compute delta in Halstead Metrics

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
    # Compute metrics with timing
    plain_logger.debug("SLOC Complexity - Summary")
    sloc_count, sloc_time = timed(sloc.compute_sloc, code)
    plain_logger.debug("\n")
    
    plain_logger.debug("Nesting Depth Complexity - Summary")
    nesting_count, nesting_time = timed(nesting_depth.compute_nesting_depth, code)
    plain_logger.debug("\n")
    
    plain_logger.debug("Cyclomatic Complexity - Summary")
    if cyclomatic_func.__name__ == "regex_compute_cyclomatic":
        cyclomatic_complexity, cyclomatic_time = timed(cyclomatic_func, code)
    else:
        cyclomatic_complexity, cyclomatic_time = timed(cyclomatic_func, code, file_path)
    plain_logger.debug("\n")
    
    plain_logger.debug("Cognitive Complexity - Summary")
    cognitive_complexity, cognitive_time = timed(cognitive.regex_compute_cognitive, code)
    plain_logger.debug("\n")
    
    plain_logger.debug("Halstead Complexity - Summary")
    halstead_metrics, halstead_time = timed(halstead.halstead_metrics, code, halstead_lang)
    plain_logger.debug("\n")
    # Log results
    metrics_logger.info("Analyzing file: %s", file_path)
    metrics_logger.info("SLOC Complexity: %d  [runtime: %.4fs]", sloc_count, sloc_time)
    metrics_logger.info("Nesting Depth Complexity: %d  [runtime: %.4fs]", nesting_count, nesting_time)
    metrics_logger.info("Cyclomatic Complexity: %d  [runtime: %.4fs]", cyclomatic_complexity, cyclomatic_time)
    metrics_logger.info("Cognitive Complexity: %d  [runtime: %.4fs]", cognitive_complexity, cognitive_time) 
    metrics_logger.info("Halstead Complexity:")
    for k, v in halstead_metrics.items():
        if k == 'op_dict':
            continue
        metrics_logger.info("  %s: %s", k, v)
    metrics_logger.info("  Vocabulary: %s", halstead.vocabulary(halstead_metrics))
    metrics_logger.info("  Size: %s", halstead.size(halstead_metrics))
    metrics_logger.info("  Volume: %.2f", halstead.volume(halstead_metrics))
    metrics_logger.info("  Difficulty: %.2f", halstead.difficulty(halstead_metrics))
    metrics_logger.info("  Effort: %.2f", halstead.effort(halstead_metrics))
    metrics_logger.info("  Time: %.2fs", halstead.time(halstead_metrics))
    metrics_logger.info("[Halstead runtime: %.4fs]", halstead_time)

    # Compute GPU delta metrics if requested
    if gpu_delta_enabled:
        baseline_metrics, _ = timed(halstead.halstead_metrics, code, "cpp")
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

def analyze_directory(directory_path: str, halstead_lang, cyclomatic_func, gpu_delta_enabled):
    """Recursively analyze all source code files in a directory.

    Args:
        directory_path (str): Path to the directory.
        halstead_func (callable): Function to compute Halstead metrics.
        gpu_delta_enabled (boolean): Flag whether to compute delta in Halstead Metrics
    """
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith((".cpp", ".cxx", ".cc", ".cu", ".cl", ".hpp", ".h")):
                file_path = os.path.join(root, file_name)
                analyze_code(file_path, halstead_lang, cyclomatic_func, gpu_delta_enabled)

def main():
    """Command-line interface for analyzing code complexity metrics.

    Accepts a file or directory path and optional arguments:
        --lang: Language for Halstead metrics
        --gpu-delta: Compare GPU constructs vs C++ baseline
        --verbose: Detailed debug logging
    """
    parser = argparse.ArgumentParser(description="Analyze code complexity metrics for a file or directory.")
    parser.add_argument("path", help="Path to the code file or directory to analyze")
    parser.add_argument("--lang", choices=[
        "cpp", "cuda", "opencl", "kokkos", "openmp","adaptivecpp", "openacc",
        "opengl_vulkan", "webgpu", "boost", "metal", "thrust", "merged", "auto"], 
    default="cpp", help="Language extension for Halstead metrics")
    parser.add_argument("--cyclomatic", choices=["advanced"], default="basic",
                    help="Cyclomatic complexity calculation method")
    parser.add_argument("--report", choices=["basic", "advanced"], default=None,
                    help="Run basic/advanced statistical analysis pipeline after computing metrics")
    parser.add_argument("--gpu-delta", action="store_true",
                    help="Compute added complexity of GPU constructs vs C++ baseline")
    parser.add_argument("--verbose", action="store_true",
                    help="Detailed debug logging") # Mostly relevant: Halstead, Cyclomatic
    args = parser.parse_args()
    
    # Enable debug logging if verbose
    if args.verbose:
        plain_logger.setLevel(logging.DEBUG)

    # Validate path
    if not os.path.exists(args.path):
        metrics_logger.error("Path does not exist: %s", args.path)
        return
    
    # Select Cyclomatic function based on method
    cyclomatic_func = {
        "advanced": cyclomatic.cfg_compute_cyclomatic,
        "basic": cyclomatic.regex_compute_cyclomatic
    }[args.cyclomatic]
    
    # Determine baseline for GPU delta
    if args.gpu_delta and args.lang != "cpp":
        gpu_delta_enabled = True
    else:
        gpu_delta_enabled = False
    
    halstead_lang = args.lang
    # Analyze file or directory
    if args.report == None:
        if os.path.isfile(args.path):
            analyze_code(args.path, halstead_lang, cyclomatic_func, gpu_delta_enabled)
        elif os.path.isdir(args.path):
            analyze_directory(args.path, halstead_lang, cyclomatic_func, gpu_delta_enabled)
        else:
            metrics_logger.error("Path is neither a file nor a directory: %s", args.path)
    # Stat Analysis
    else:
        report_func = {
            "basic": generate_basic_report,
            "advanced": generate_advanced_report
        }[args.report]
        stat_report(args.path, report_func)
    
if __name__ == "__main__":
    main()