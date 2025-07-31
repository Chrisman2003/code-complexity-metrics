import argparse
import os 
from projectFolder.metrics import sloc, halstead, cyclomatic

def analyze_code(file_path: str):
    with open(file_path, 'r') as file:
        code = file.read()        
    sloc_count = sloc.compute_sloc(code)

    print(f"File: {file_path}")
    print(f"SLOC: {sloc_count}")
    
def main():
    parser = argparse.ArgumentParser(description="Analyze code complexity metrics.")
    parser.add_argument("file", help="Path to the code file to analyze")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: {args.file} is not a valid file.")
        return

    analyze_code(args.file)