#!/usr/bin/env python3
import argparse

def filter_nesting_depth_lines_from_file(filename: str) -> None:
    """Overwrite the file with only lines containing 'Nesting Depth'."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    filtered_lines = [line for line in lines if "Nesting Depth" in line]

    # Overwrite the file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)

def main():
    parser = argparse.ArgumentParser(
        description="Keep only lines containing 'Nesting Depth' in a file (overwrites file)."
    )
    parser.add_argument("file", type=str, help="Path to the input file to filter.")
    args = parser.parse_args()

    filter_nesting_depth_lines_from_file(args.file)

if __name__ == "__main__":
    main()
