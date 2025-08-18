import argparse
import os
import time
from projectFolder.metrics import sloc, halstead, cyclomatic

# Timed execution wrapper
def timed(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

# Analyze a single code file
def analyze_code(file_path: str, halstead_func, gpu_baseline_func=None):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
        code = file.read()

    # Compute metrics with timings
    sloc_count, sloc_time = timed(sloc.compute_sloc, code)
    halstead_metrics, halstead_time = timed(halstead_func, code)
    cyclomatic_complexity, cyclomatic_time = timed(cyclomatic.basic_compute_cyclomatic, code)

    print(f"File: {file_path}")
    print(f"SLOC: {sloc_count} \n[SLOC runtime: {sloc_time:.4f}s]")

    print(f"\nCyclomatic Complexity: {cyclomatic_complexity} \n[Cyclomatic runtime: {cyclomatic_time:.4f}s]")

    print("\nHalstead Metrics:")
    for k, v in halstead_metrics.items():
        print(f"  {k}: {v}")
    print(f"  Vocabulary: {halstead.vocabulary(halstead_metrics)}")
    print(f"  Size: {halstead.size(halstead_metrics)}")
    print(f"  Volume: {halstead.volume(halstead_metrics):.2f}")
    print(f"  Difficulty: {halstead.difficulty(halstead_metrics):.2f}")
    print(f"  Effort: {halstead.effort(halstead_metrics):.2f}")
    print(f"  Time: {halstead.time(halstead_metrics):.2f}s")
    print(f"[Halstead runtime: {halstead_time:.4f}s]")

    # Compute GPU delta if requested
    if gpu_baseline_func:
        baseline_metrics, _ = timed(gpu_baseline_func, code)
        delta = {k: halstead_metrics[k] - baseline_metrics[k] for k in halstead_metrics}
        print("\nGPU Delta Metrics (GPU - C++ baseline):")
        for k, v in delta.items():
            print(f"  {k}: {v}")
        print(f"  Vocabulary delta: {halstead.vocabulary(halstead_metrics) - halstead.vocabulary(baseline_metrics)}")
        print(f"  Size delta: {halstead.size(halstead_metrics) - halstead.size(baseline_metrics)}")
        print(f"  Volume delta: {halstead.volume(halstead_metrics) - halstead.volume(baseline_metrics):.2f}")
        print(f"  Difficulty delta: {halstead.difficulty(halstead_metrics) - halstead.difficulty(baseline_metrics):.2f}")
        print(f"  Effort delta: {halstead.effort(halstead_metrics) - halstead.effort(baseline_metrics):.2f}")
        print(f"  Time delta: {halstead.time(halstead_metrics) - halstead.time(baseline_metrics):.2f}s")
    
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Analyze code complexity metrics.")
    parser.add_argument("file", help="Path to the code file to analyze")
    parser.add_argument("--lang", choices=["cuda", "opencl", "kokkos", "cpp"], default="cpp",
                        help="Language extension for Halstead metrics")
    parser.add_argument("--gpu-delta", action="store_true",
                        help="Compute added complexity of GPU constructs vs C++ baseline")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: {args.file} is not a valid file.")
        return

    # Select Halstead function based on language
    if args.lang == "cuda":
        halstead_func = halstead.halstead_metrics_cuda
    elif args.lang == "opencl":
        halstead_func = halstead.halstead_metrics_opencl
    elif args.lang == "kokkos":
        halstead_func = halstead.halstead_metrics_kokkos
    else:
        halstead_func = halstead.halstead_metrics_cpp

    # Determine baseline for GPU delta
    gpu_baseline_func = None
    if args.gpu_delta and args.lang != "cpp":
        gpu_baseline_func = halstead.halstead_metrics_cpp

    analyze_code(args.file, halstead_func, gpu_baseline_func=gpu_baseline_func)

if __name__ == "__main__":
    main()
