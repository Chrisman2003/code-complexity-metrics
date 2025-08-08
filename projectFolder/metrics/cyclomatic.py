def basic_compute_cyclomatic(code: str) -> int:
    # C++ control flow keywords and logical operators
    control_keywords = [
        'if', 'for', 'while', 'case', 'catch', 'switch', 'else', 'do', 'goto'
    ]
    logical_operators = ['&&', '||', '?', 'and', 'or']  # 'and', 'or' for alternative tokens
    count = 0 
    for line in code.splitlines():
        stripped = line.strip()
        # Count control keywords at the start of the line (simple heuristic)
        for keyword in control_keywords:
            if stripped.startswith(keyword):
                count += 1
        # Count logical operators anywhere in the line
        for op in logical_operators:
            count += stripped.count(op)
    return count + 1  # +1 for


