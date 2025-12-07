# Testing for `code_complexity` package
This `testing` package contains unit tests for the `code_complexity` Python package. The tests are written using **pytest** and cover all major functionality of the `metrics` subpackage, including Cognitive, Cyclomatic, Halstead, Nesting Depth and SLOC complexity. Parallel framework detection as well as utility functions are likewise tested.

These tests help ensure the correctness and reliability of the metrics computations and preprocessing tools. 
Their primary aim is to detect whether changes in code functionality lead to different code complexity metric values. As specified in the thesis, the testing pipeline is automated with the corresponding GPT model. Manual demonstrations are displayed in the thesis and performed on the directory 'samples/thesis_testing/manual'.
The pytest system is linked with the CI pipeline, such that they are automatically run upon a pull / push request.