import argparse
import os 
import time
from projectFolder.metrics import sloc, halstead, cyclomatic
from projectFolder.gpu import halstead_opencl

# Outsourced Time analyzing Function
def timed(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def analyze_code(file_path: str):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
        code = file.read()        
    sloc_count, sloc_time = timed(sloc.compute_sloc, code)
    halstead_metrics, halstead_time = timed(halstead.basic_halstead_metrics, code)
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
    print("-" * 40)
    
def analyze_code_cpu(file_path: str):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
        code = file.read()
    # Computing Tokens as Preprocessing
    halstead_metrics = halstead.basic_halstead_metrics(code)
    halstead_opencl.gpu_compute_halstead(halstead_metrics)
    
    print(f"File: {file_path}")
    print("\nHalstead Metrics (GPU):")
    for k, v in halstead_metrics.items():
        print(f"  {k}: {v}")
    print(f"  Vocabulary: {halstead.vocabulary(halstead_metrics)}")
    print(f"  Size: {halstead.size(halstead_metrics)}")
    print(f"  Volume: {halstead.volume(halstead_metrics):.2f}")
    print(f"  Difficulty: {halstead.difficulty(halstead_metrics):.2f}")
    print(f"  Effort: {halstead.effort(halstead_metrics):.2f}")
    print(f"  Time: {halstead.time(halstead_metrics):.2f}s")
    
def main():
    parser = argparse.ArgumentParser(description="Analyze code complexity metrics.")
    parser.add_argument("file", help="Path to the code file to analyze")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for Halstead metrics computation")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: {args.file} is not a valid file.")
        return
    
    if args.gpu:
        analyze_code_cpu(args.file)
    else:
        analyze_code(args.file)

if __name__ == "__main__":
    main()