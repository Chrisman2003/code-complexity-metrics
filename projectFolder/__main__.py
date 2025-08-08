import argparse
import os 
from projectFolder.metrics import sloc, halstead, cyclomatic

def analyze_code(file_path: str):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
        code = file.read()        
    sloc_count = sloc.compute_sloc(code)
    halstead_metrics = halstead.basic_halstead_metrics(code)
    cyclomatic_complexity = cyclomatic.basic_compute_cyclomatic(code)    
    print(f"File: {file_path}")
    print(f"SLOC: {sloc_count}")
    print("\nHalstead Metrics:")
    for k, v in halstead_metrics.items():
        print(f"  {k}: {v}")
    print(f"  Vocabulary: {halstead.vocabulary(halstead_metrics)}")
    print(f"  Size: {halstead.size(halstead_metrics)}")
    print(f"  Volume: {halstead.volume(halstead_metrics):.2f}")
    print(f"  Difficulty: {halstead.difficulty(halstead_metrics):.2f}")
    print(f"  Effort: {halstead.effort(halstead_metrics):.2f}")
    print(f"  Time: {halstead.time(halstead_metrics):.2f}s")

    print(f"\nCyclomatic Complexity: {cyclomatic_complexity}")
    
def main():
    parser = argparse.ArgumentParser(description="Analyze code complexity metrics.")
    parser.add_argument("file", help="Path to the code file to analyze")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: {args.file} is not a valid file.")
        return

    analyze_code(args.file)