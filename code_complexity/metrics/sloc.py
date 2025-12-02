# -----------------------------------------------------------------------------
# Source Lines of Code (SLOC) Computation for C++ and GPU-Extended Code
# -----------------------------------------------------------------------------
# Includes:
# - Counts all non-empty source lines after removing comments and string literals
# - Works for C++ and GPU-extended languages (CUDA, OpenCL, SYCL, OpenMP, etc.)
#
# Note:
# This module provides an accurate SLOC metric by ignoring non-executable text.
# Only lines containing actual executing code contribute to the count.
# -----------------------------------------------------------------------------
from code_complexity.metrics.utils import *

def compute_sloc(code: str) -> int:
    """Compute Source Lines of Code (SLOC) after removing comments and string literals.

    Counts all non-empty lines that remain after stripping comments and (non-kernel) string literals.

    Args:
        code (str): Full C++ extended source code.

    Returns:
        int: Number of source lines of code (SLOC).
    """
    # Remove comments, non-kernel string literals, Header calls
    plain_logger.debug(f"Initial code size: {len(code.splitlines())} lines")
    
    # Before/after header removal
    before = sum(1 for line in code.splitlines() if line.strip())
    code = remove_headers(code)
    after = sum(1 for line in code.splitlines() if line.strip())
    plain_logger.debug(f"Header removal reduced lines by: {before - after}")
    
    # Before/after comment removal
    before = sum(1 for line in code.splitlines() if line.strip())
    code = remove_cpp_comments(code)
    after = sum(1 for line in code.splitlines() if line.strip())
    plain_logger.debug(f"Comment removal reduced lines by: {before - after}")
    
    # Before/after string literal removal
    before = sum(1 for line in code.splitlines() if line.strip())
    code = remove_string_literals(code)
    after = sum(1 for line in code.splitlines() if line.strip())
    plain_logger.debug(f"String literal reduced lines by: {before - after}")
    
    #code = remove_headers(code)
    #code = remove_cpp_comments(code)
    #code = remove_string_literals(code)

    # Count all non-empty lines (original and those from removeals)
    lines = code.splitlines()
    return sum(1 for line in lines if line.strip())


"""EDGE CASE:
String literals:

const char* msg = "line1
line2
line3";
Without removal, this counts as 3 lines.

With removal, it becomes:
const char* msg = ;
"""