# Code Complexity Analyzer
A pip installable Python tool to analyze source code and compute: **SLOC**, **Cyclomatic**, **Cognitive**, and **Halstead** Code Complexity Measures. The tool supports parallelizing frameworks for code constructs and particularly with respect to Halstead metrics may specifically measure these constructs specifically in addition to the cpp base code.

### Code Complexity Measures:
- SLOC
- Nested Depth
- Cyclomatic
- Cognitive
- Halstead

### 11 Parallelizing Frameworks on C++:
- CUDA
- OpenCL
- Kokkos
- OpenMP 
- AdaptiveCPP
- OpenACC
- OpenGlVulkan 
- WebGPU
- Boost
- Metal
- Thrust

The Program is Target Agnostic, hence one needs to merely type the high-level prepackaged "code-metrics" command with the relative location of some Directory or File on your machine. Typically your personal directories will be outside this folder by some hierarchy levels. Therefore corresponding prefixation with "../" to the required depth should be applied.

### System Installation
- Aside from the auto configured pip install packages a system insall of clang-18 is needed for version compatibility
1) [Unrecommended] 
sudo apt install clang-18 lldb-18 lld-18 
1) [Recommended]: Installs LLVM/Clang directly from official LLVM APT repositories.
``bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18
``
2) Add LLVM Libraries to the system linker:
- echo "/usr/lib/llvm-18/lib" | sudo tee /etc/ld.so.conf.d/llvm-18.conf
- sudo ldconfig
3) If your system install has unconventional name formatting: create a symlink
- sudo ln -s /usr/lib/llvm-18/lib/"libclang-18.so.1" /usr/lib/llvm-18/lib/libclang.so
- [Where "libclang-18.so.1" is YOUR own malformatted name string!]
- [This Creates a shortcut: It makes a new file entry at the target path, which is /usr/lib/llvm-18/lib/libclang.so]
- [Points to the actual file: This new entry acts as a pointer or alias to the existing, specific 
- versioned file: /usr/lib/llvm-18/lib/libclang-18.so.1]
- sudo ldconfig

## How It Works
- The tool analyzes a source code file or directory in four main steps:

### 1. Preprocessing and Regex
- Reads the source code file (e.g., `main.cpp`).  
- Performs Regex Analysis with Word Boundaries
- Ignores comments, whitespace, and string literals to prevent interference with metric calculations.  

### 2. Cyclomatic Complexity
- Specification standard pursuant to Thomas J McCabe
- T. J. McCabe, "A Complexity Measure," in IEEE Transactions on Software Engineering, vol. SE-2,

### 3. Cognitive Complexity
- Specification standard pursuant to Sonarqube
- {Cognitive Complexity} a new way of measuring understandability By G. Ann Campbell

### 4. Halstead Metrics
- Counts **operators** and **operands**, both total and unique.  
- Computes:  
  - Vocabulary: η = η₁ + η₂
  - Program Size: N = N₁ + N₂
  - Volume: V = N × log₂(η)
  - Difficulty: D = (η₁ / 2) × (N₂ / η₂)
  - Effort: E = V × D
  - Time: T = E/k
  In the above formulae, k is the stroud number, which has an arbitrary default value of 18.
  https://www.ibm.com/docs/en/devops-test-embedded/9.0.0?topic=metrics-halstead
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



