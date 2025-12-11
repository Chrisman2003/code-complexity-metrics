## Directory of sample files used for pytests, metric verification, and thesis result analysis
Some of the files are auto-generated to analyze for metric complexity. This is in correspondence with the objective thesis claim of solely verifying manual correctness for the files in the subdirectory `thesis_testing/manual`. Pytests run on other sample files are verified with the cited GPT-model in the thesis text. This GPT-model is likewise used to actually create files to analyze under a certain language/framework. For files in a given parallel framework, files from the linked GitHub repositories may be used to prime the GPT engine before it is used to auto-generate further files within the parallel framework to test.

NOTE: not all files (especially those of manual testing) are semantically correct. This is due to code complexity metrics being a tool for static code analysis. Other than the Clang Cyclomatic complexity implementation, no other code complexity implementation requires semantically correct files.

## cpp
- **OLD_simple.cpp** – Old, simple C++ example, a basic starting point.
- **case.cpp** – Sample C++ code to test control flow and case statements.
- **edge.cpp** – C++ code testing edge cases for metrics or parsing.
- **edge2.cpp** – Additional edge case C++ code.
- **isolated.cpp** – Small, self-contained C++ snippet for isolated testing.
- **new.cpp** – Newer C++ example with modern constructs.
- **new2.cpp** – Another modern C++ example similar to `new.cpp`.
- **complex.cpp** – C++ code with multiple features to test code complexity metrics.
- **hyper_complex.cpp** – Very complex C++ sample with nested structures and advanced logic.

## cuda
- **advanced.cu** – CUDA code with more advanced operations than basic examples.
- **advanced_2.cu** – Another advanced CUDA example with variations from the first.
- **complex.cu** – CUDA implementation of complex computations.
- **edge1.cu** – CUDA example focusing on edge cases.
- **edge2.cu** – Another CUDA edge case sample.
- **edge3.cu** – Third CUDA edge case variant.
- **matrix_mul.cu** – CUDA implementation of matrix multiplication.
- **matrix_mul2.cu** – Alternative version of matrix multiplication in CUDA.
- **online.cu** – CUDA code intended from an online source for demonstration purposes.

## kokkos
- **advanced.cpp** – Advanced C++ example using Kokkos for parallelism.
- **advanced2.cpp** – Another advanced Kokkos sample with different patterns.
- **complex.cpp** – Kokkos-based parallel version of complex C++ code.

## opencl
- **complex.cpp** – OpenCL version of the complex C++ sample.
- **complex2.cpp** – Alternative OpenCL implementation with slight variations.
- **edge_cases.cpp** – OpenCL code designed to test edge cases in parsing or execution.

## adaptivecpp, boost, metal, openacc, opengl_vulkan, openmp, thrust, webgpu
- **simple.cpp** - simple program for analysis with the framework
- **complex.cpp** - complex program for analysis with the framework

## thesis_testing
- manual - this subdirectory represents all of the files for the Thesis text, which were used for manual demonstrations in order to verify correctness of code complexity metric implementations.
- ANALYSIS_REPOSITORY - this subdirectory contains the sets of files, which were used for evaluating algorithmic and framework complexity respectively. The algorithms in question are a: polyhedral gravity model, n-body simulation, matrix multiplication and vector addition.
- sudoku.cpp - file used for inter-metric code complexity analysis.
Sources [Implementations]:
- https://github.com/schuhmaj/performance-portability-benchmark (last accessed: 15 November 2025)
- https://github.com/schuhmaj/polyhedral-gravity-model-parallel (last accessed: 15 November 2025)