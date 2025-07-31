from mainFolder.metrics.sloc import compute_sloc
assert compute_sloc("print('Hello')\nprint('World')") == 2
