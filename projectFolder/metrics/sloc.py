def compute_sloc(code: str) -> int:
    lines = code.splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
    