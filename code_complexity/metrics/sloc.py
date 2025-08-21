def compute_sloc(code: str) -> int:
    """Compute the Source Lines of Code (SLOC) for C++ code.

    Counts all non-empty, non-comment lines. Handles:
    - Single-line comments (`//`)
    - Multi-line comments (`/* ... */`)
    - Preprocessor directives are counted as lines of code

    Args:
        code (str): The C++ source code.

    Returns:
        int: Number of source lines of code.
    """
    lines = code.splitlines()
    count = 0
    in_block_comment = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Handle multi-line comments
        while True:
            if in_block_comment:
                if '*/' in stripped:
                    in_block_comment = False
                    stripped = stripped.split('*/', 1)[1].strip()
                    continue  # recheck remaining line
                else:
                    stripped = ''
                    break
            elif '/*' in stripped:
                in_block_comment = True
                stripped = stripped.split('/*', 1)[0].strip()
                continue  # recheck remaining line
            break

        # Skip single-line comments
        if stripped.startswith('//') or not stripped:
            continue

        count += 1

    return count