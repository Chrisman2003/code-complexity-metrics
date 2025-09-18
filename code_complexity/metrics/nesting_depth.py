def compute_nesting_depth(code: str) -> int:
    """Compute the maximum nesting depth for C++ code.

    Counts the deepest level of nested scopes based on curly braces `{}`.
    Ignores braces that appear inside comments or string literals.

    Args:
        code (str): The C++ source code.

    Returns:
        int: Maximum nesting depth.
    """
    lines = code.splitlines()
    max_depth = 0
    current_depth = 0
    in_block_comment = False
    in_string = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        i = 0
        while i < len(stripped):
            char = stripped[i]
            # Handle string literals
            # Checks if we’re entering/exiting a string ("..." or '...').
            # { and } inside strings should not count toward nesting
            if not in_block_comment and char in ('"', "'"):
                if not in_string:
                    in_string = char
                elif in_string == char:
                    in_string = False
                i += 1
                continue

            # Handle multi-line comments
            if not in_string:
                if in_block_comment:
                    if '*/' in stripped[i:]:
                        end = stripped.find('*/', i)
                        i = end + 2
                        in_block_comment = False
                        continue
                    else:
                        break  # rest of line is still in comment
                elif stripped.startswith('/*', i):
                    in_block_comment = True # entering block comment
                    i += 2
                    continue

                # Skip single-line comments
                if stripped.startswith('//', i):
                    break

                # Count nesting depth based on { and } with respect to control flow structures
                if char == '{':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == '}':
                    current_depth = max(0, current_depth - 1)

            i += 1

    return max_depth