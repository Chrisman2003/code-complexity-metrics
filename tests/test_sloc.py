from projectFolder.metrics.sloc import compute_sloc

def test_sloc_basic():
    assert compute_sloc("") == 0
    assert compute_sloc("   ") == 0
    assert compute_sloc("# This is a comment") == 0
    assert compute_sloc("print('Hello')") == 1
    assert compute_sloc("# Comment\nprint('Hello')") == 1
    assert compute_sloc("print('Hello')\n# Another comment") == 1
    assert compute_sloc("# Comment\n# Another comment") == 0
