def naive_compute_cyclomatic(code: str) -> int:
    control_keywords = ['if', 'for', 'while', 'elif', 'case', 'catch', 'except', 'finally', 'with']
    logical_operators = ['&&', '||', 'and', 'or']    
    count = 0 
    for line in code.splitlines():
        stripped = line.strip()
        for keyword in control_keywords:
            if stripped.startswith(keyword):  # more precise
                count += 1
        for op in logical_operators:
            count += stripped.count(op)  # handle multiple per line
    return count + 1  # +1 for default path


