# Code Complexity Analyzer

A Python tool to analyze source code and compute Code Complexity Measures.**SLOC**, **Cyclomatic Complexity**, **Cognitive Complexity**, and **Halstead Metrics**. The tool supports parallelizing frameworks for code constructs and particularly with respect to halstead metrics may specifically measure these constructs specifically in addition to the cpp base code.
-> Code Complexity Measures:
--> SLOC
--> Nested Depth
--> Cyclomatic
--> Cognitive
--> Halstead

-> Parallelizing Frameworks:
--> C++
--> CUDA 
--> OpenCL
--> Kokkos
--> (SyCl)
--> (OpenMP)

The Program is Target Agnostic, hence one needs to merely type the high-level prepackaged "code-metrics" command with the location of some Directory or File on your machine. Typically your personal directories will be outside this folder by some hierarchy levels. Therefore corresponding prefixation with "../" to the required depth should be applied.

---

## Features

- **SLOC (Source Lines of Code)**: Measures the number of actual code lines, ignoring comments and whitespace.  
- **Cyclomatic Complexity**: Counts the number of linearly independent paths in a program.  
- **Cognitive Complexity**: Estimates the effort required to understand the code, accounting for control flow and nesting.  
- **Halstead Metrics**: Measures code complexity based on operators and operands, including vocabulary, program length, volume, difficulty, and effort.  
- **GPU Delta Metrics**: Compare the added complexity of GPU constructs against a C++ baseline.  

---

## How It Works

The tool analyzes a source code file or directory in four main steps:

### 1. Preprocessing and Tokenization
- Reads the source code file (e.g., `main.cpp`).  
- Breaks the code into tokens (words, symbols, numbers).  
- Ignores comments, whitespace, and string literals to prevent interference with metric calculations.  

### 2. Cyclomatic Complexity
- Initializes complexity to 1 (single path).  
- Iterates through tokens and increments complexity for control structures like `if`, `for`, `while`, `case`, and `catch`.  

### 3. Cognitive Complexity
- Initializes complexity to 0.  
- Adds 1 for each control flow keyword (`if`, `for`, `while`, `switch`).  
- Adds 1 for each level of nesting inside loops or conditional blocks.  

### 4. Halstead Metrics
- Counts **operators** and **operands**, both total and unique.  
- Computes:  
  - Program Length: `N = N₁ + N₂`  
  - Vocabulary: `η = η₁ + η₂`  
  - Volume: `V = N × log₂(η)`  
  - Difficulty: `D = (η₁ / 2) × (N₂ / η₂)`  
  - Effort: `E = V × D`  

---

## Future Plans
- Improved cyclomatic and cognitive complexity analysis for GPU constructs
- More detailed Halstead metrics distinguishing different operand types
- Comprehensive pytest suite for accuracy verification

## Installation
- Make sure Python 3.8+ is installed.
- Future Cyclomatic and Cognitive Complexity will require the following 
- SUDO apt dependencies: 
- sudo apt install kokkos libkokkos-dev
- sudo apt install libomp-dev
- sudo apt install nvidia-cuda-toolkit
- sudo apt install clang-15 llvm-15
- SDK kit for cuda



