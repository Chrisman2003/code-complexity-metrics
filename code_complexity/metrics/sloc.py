from code_complexity.metrics.shared import *

def compute_sloc(code: str) -> int:
    """Compute Source Lines of Code (SLOC) after removing comments and string literals.

    Counts all non-empty lines that remain after stripping comments and (non-kernel) string literals.

    Args:
        code (str): Full C/C++/OpenCL source code.

    Returns:
        int: Number of source lines of code (SLOC).
    """
    # Remove all Header Calling Instances
    code = remove_headers(code)
    # Remove all comments
    code = remove_cpp_comments(code)
    # Remove all string literals except kernel strings
    code = remove_string_literals(code)
    # Count all non-empty lines
    lines = code.splitlines()
    return sum(1 for line in lines if line.strip())
