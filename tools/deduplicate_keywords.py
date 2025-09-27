"""Helper script to deduplicate keyword definitions in a Python file.

This script scans for single-quoted words (e.g., `'for'`, `'if'`, `'while'`) in
the specified file, removes duplicates, and rewrites the file with a deduplicated
set of keywords. It also prints a summary of which words were removed and on which
lines they occurred.

Example:
    Run this script from the command line to deduplicate a file:

        python deduplicate_keywords.py KEYWORDS.py
"""

import re
import sys
from pathlib import Path


def deduplicate_keywords(file_path: str):
    """Deduplicate single-quoted keywords in a file.

    Reads a Python file, identifies all single-quoted keywords (e.g. `'for'`),
    and removes duplicates while preserving the first occurrence. The cleaned
    result is written back to the same file.

    Args:
        file_path (str): Path to the file that should be deduplicated.

    Raises:
        SystemExit: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")

    seen = set()  # Tracks keywords we have already encountered
    removed_words = []  # Stores (word, line_number) for reporting
    result_lines = []  # Stores the deduplicated lines for rewriting

    # Pattern to match single-quoted words inside sets or lists
    pattern = re.compile(r"'([^']+)'")

    for line_num, line in enumerate(text.splitlines(), start=1):
        new_line = line
        matches = pattern.findall(line)

        for match in matches:
            if match in seen:
                # Record this duplicate for reporting
                removed_words.append((match, line_num))
                # Remove the duplicate keyword (with possible trailing comma)
                new_line = re.sub(
                    rf"\s*'{re.escape(match)}'\s*,?", "", new_line, count=1
                )
            else:
                # First time we've seen this keyword, keep it
                seen.add(match)

        # Clean up any leftover commas or trailing commas before closing brace
        new_line = re.sub(r",\s*,", ",", new_line)
        new_line = re.sub(r",\s*}", " }", new_line)

        result_lines.append(new_line)

    # Overwrite the file with the deduplicated version
    path.write_text("\n".join(result_lines), encoding="utf-8")

    # Print summary of what was removed
    if removed_words:
        print(f"✅ Deduplicated {file_path}. Removed {len(removed_words)} duplicates:")
        for word, line in removed_words:
            print(f"  - '{word}' (line {line})")
    else:
        print(f"✅ No duplicates found in {file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python deduplicate_keywords.py <file-to-deduplicate>")
        sys.exit(1)

    deduplicate_keywords(sys.argv[1])
