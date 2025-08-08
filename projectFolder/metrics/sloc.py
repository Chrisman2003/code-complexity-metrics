def compute_sloc(code: str) -> int:
    lines = code.splitlines()
    count = 0
    in_block_comment = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if in_block_comment:
            if '*/' in stripped:
                in_block_comment = False
                # Remove up to and including */
                stripped = stripped.split('*/', 1)[1].strip()
                if not stripped:
                    continue
            else:
                continue
        # Handle start of block comment
        if '/*' in stripped:
            in_block_comment = True
            # Remove from /* onwards
            stripped = stripped.split('/*', 1)[0].strip()
            if not stripped:
                continue
        # Skip single-line comments
        if stripped.startswith('//'):
            continue
        if stripped:
            count += 1
    return count
