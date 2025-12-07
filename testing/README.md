# Testing for `code_complexity` package
This `testing` package contains unit tests for the `code_complexity` Python package. The tests are written using **pytest** and cover all major functionality of the `metrics` subpackage, including Cognitive, Cyclomatic, Halstead, Nesting Depth and SLOC complexity. Parallel framework detection as well as utility functions are likewise tested.

These tests help ensure the correctness and reliability of the metrics computations and preprocessing tools. 
The pytest system is linked with the CI pipeline, such that they are automatically run upon a pull / push request.