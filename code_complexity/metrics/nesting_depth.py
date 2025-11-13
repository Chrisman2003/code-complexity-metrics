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

    for char in code:
        if char == '{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == '}':
            current_depth = max(0, current_depth - 1)

    return max_depth