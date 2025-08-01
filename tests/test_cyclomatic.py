from projectFolder.metrics.cyclomatic import naive_compute_cyclomatic

# Testing Code Complexity = Structures + 1 (Default Path)
def test_no_branches():
    code = """
def foo():
    return 42
"""
    # No control structures, so complexity = 0 + 1
    assert naive_compute_cyclomatic(code) == 1

def test_single_if():
    code = """
def foo(x):
    if x > 0:
        return x
    return -x
"""
    # One 'if' structure, so complexity = 1 + 1
    assert naive_compute_cyclomatic(code) == 2

def test_multiple_branches():
    code = """
def foo(x):
    if x > 0:
        return x
    elif x == 0:
        return 0
    else:
        return -x
"""
    # 'if', 'elif' are counted as branches, so complexity = 2 (if, elif) + 1
    assert naive_compute_cyclomatic(code) == 3

def test_loops_and_try():
    code = """
def bar(lst):
    for x in lst:
        try:
            print(x)
        except Exception:
            pass
"""
    # 'for', 'try', 'except' are counted, so complexity = 2 (for, try) + 1 (except) + 1
    assert naive_compute_cyclomatic(code) == 4

def test_logical_operators():
    code = """
def baz(a, b):
    if a && b:
        return True
    return False
"""
    # 'if' and 'and' are counted, so complexity = 2 + 1
    assert naive_compute_cyclomatic(code)