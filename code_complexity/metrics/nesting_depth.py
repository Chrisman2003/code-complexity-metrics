# -----------------------------------------------------------------------------
# Nesting Depth Analysis for C++ and OpenCL Code
# -----------------------------------------------------------------------------
# Includes:
# - Maximum scope nesting computation based on curly braces
# - Works for C++ and GPU-extended code (OpenCL, CUDA, SYCL, etc.)
#
# Note:
# This module computes the maximum nesting depth of a code block, which is
# useful for understanding code complexity and potential maintainability issues.
# -----------------------------------------------------------------------------
from code_complexity.metrics.utils import *

def compute_nesting_depth(code: str) -> int:
    """Compute the maximum nesting depth for C++/OpenCL code.

    This function removes comments and string literals before
    counting curly braces to determine the maximum depth of nested scopes.

    Args:
        code (str): The source code.

    Returns:
        int: Maximum nesting depth.
    """
    # Remove comments, non-kernel string literals, Header calls
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code = remove_string_literals(code)

    max_depth = 0
    current_depth = 0
    line_num = 1

    for char in code:
        if char == '\n':
            line_num += 1
        if char == '{':
            current_depth += 1
            if current_depth > max_depth:
                plain_logger.debug(
                f"New max nesting depth {current_depth} reached at line {line_num}"
            )
            max_depth = max(max_depth, current_depth)
        elif char == '}':
            current_depth = max(0, current_depth - 1)
    return max_depth