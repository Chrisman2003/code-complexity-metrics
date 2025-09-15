## Step 1: Preprocessing and Tokenization
The tool starts by reading the target source code file (e.g., `complex.cpp`). It then tokenizes the code, breaking it down into a list of individual words, symbols, and numbers. This process handles comments, whitespace, and string literals, ensuring they don't interfere with the analysis.

## Step 2: Cyclomatic Complexity Calculation (Basic Version)
The initial phase of complexity analysis focuses on Cyclomatic Complexity. This metric measures the number of linearly independent paths through a program's source code. The algorithm identifies specific keywords and control structures that contribute to the complexity score.

- Initialize complexity to 1 (for the single path through the function).  
- Iterate through the tokenized list of the code.  
- Increment the complexity counter for each occurrence of the following keywords:  
  `if`, `for`, `while`, `case`, `catch`

The total count represents the basic cyclomatic complexity of the code.

## Step 3: Cognitive Complexity Calculation (Basic Version)
This metric is designed to measure the effort required for a developer to understand a piece of code. The current, basic implementation follows a simple additive model.

- Initialize cognitive complexity to 0.  
- Iterate through the tokens of the code.  
- Add a complexity score for each of the following:  
  - **Control Flow Keywords:** Add 1 for each `if`, `for`, `while`, `switch`.  
  - **Nesting:** Add 1 for each level of nesting within a loop or conditional block.

The final sum represents the basic cognitive complexity.

## Step 4: Halstead Metrics Calculation
The project also computes Halstead Metrics, which are based on counting operators and operands in the source code.

- Count unique operators (η₁) and total operators (N₁).  
- Count unique operands (η₂) and total operands (N₂).

These four values are then used to calculate various Halstead metrics, such as:

- **Program Length:** `N = N₁ + N₂`  
- **Vocabulary Size:** `η = η₁ + η₂`  
- **Volume:** `V = N × log₂(η)`  
- **Difficulty:** `D = (η₁ / 2) × (N₂ / η₂)`  
- **Effort:** `E = V × D`

## 🗺️ Future Plans
This project is under continuous development. The following features are planned for future releases:

- **Advanced Cyclomatic and Cognitive Complexity:**  
  The current implementation uses a basic keyword set. Future plans involve expanding the keyword list to include GPGPU-specific constructs from OpenCL, CUDA, and Kokkos. This will allow for more accurate complexity analysis of heterogeneous code.

- **Expanded GPGPU Language Support:**  
  The goal is to provide comprehensive support for CUDA and Kokkos in addition to OpenCL. This includes a more refined tokenization process that recognizes specific GPGPU-related data types and functions.

- **Refined Halstead Metrics:**  
  The Halstead metric calculation will be updated to handle GPGPU-specific operators and to provide a more nuanced analysis that distinguishes between different types of operands.

- **Comprehensive Testing:**  
  The current `pytest` suite is being updated. Future work will include a more extensive set of test cases to ensure the accuracy and reliability of all metric calculations across different code bases and language extensions.

- **Updated Dependencies:**  
  Once the advanced implementation is complete, the following system dependencies will be required for full functionality:  
  `clinfo`, `libviennacl-dev`, `libboost-all-dev`, `g++`, `libjsoncpp-dev`, `ocl-icd-opencl-dev`, `clc`, `libglew-dev`.
