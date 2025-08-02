from projectFolder.metrics.halstead import *

def test_if_statement():
    code = """
if x > 0:
    y = x
else:
    y = -x
"""
    metrics = basic_halstead_metrics(code)
    assert metrics['n1'] == 6
    assert metrics['n2'] == 3      # x, y, 0
    assert metrics['N1'] == 8      # all 6 above + '=' again
    assert metrics['N2'] == 6      # x, 0, y, x, y, x
    v = volume(metrics)
    d = difficulty(metrics)
    e = effort(metrics)
    t = time(metrics)
    assert v >= 0
    assert d >= 0
    assert e >= 0
    assert t >= 0