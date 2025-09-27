import networkx as nx
import subprocess
import tempfile
import os
import clang.cindex
import sys
from code_complexity.metrics.cyclomatic import get_clang_include_flags
from code_complexity.metrics.cyclomatic import build_cfg_from_dump
# -----------------------------------------------------------------------------
# Cyclomatic Complexity Analysis for C++ Code
# -----------------------------------------------------------------------------
# This module computes the McCabe cyclomatic complexity (V(G) = E − N + 2P)
# for C++ code using Clang's static analyzer for precise CFG extraction,
# as well as a simpler heuristic method based on control-flow keywords.
# -----------------------------------------------------------------------------
def compute_cyclomatic(code: str, filename: str) -> int:
    """Compute cyclomatic complexity of C++ code using Clang CFG.

    Uses Clang's static analyzer to dump the control-flow graph (CFG) of each
    function. Cyclomatic complexity is then computed as:
    
        V(G) = E - N + 2P

    where:
        E = number of edges in the CFG
        N = number of nodes in the CFG
        P = number of connected components (always 1 per function)

    Args:
        code (str): A string containing C++ source code.
        filename (str): The filename (used to determine if CUDA).

    Returns:
        int: Total cyclomatic complexity across all functions.
    """
    suffix = ".cu" if filename.endswith(".cu") else ".cpp"
    
    # Ensure the temp file is created where the original file lives so quoted includes ("...") resolve.
    original_path = os.path.abspath(filename)
    original_dir = os.path.dirname(original_path) if os.path.exists(original_path) else os.getcwd()
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8", dir=original_dir) as tmp:
            tmp_path = tmp.name
            if suffix == ".cu":
                # Wrap the entire code so that any definition of these macros is neutralized
                wrapped_code = (
                    "#ifdef __noinline__\n#undef __noinline__\n#endif\n"
                    "#ifdef __forceinline__\n#undef __forceinline__\n#endif\n" + 
                    code  # Original user code comes after
                )
                tmp.write(wrapped_code)
            else:
                tmp.write(code)
        include_flags = get_clang_include_flags()
        # Add the source directory and its parent (samples/project root) to Clang -I so local headers are found.
        project_sample_dir = os.path.abspath(os.path.join(original_dir, ".."))  # e.g. repo/samples
        # Deduplicate and extend include flags
        extra_includes = []
        for p in (original_dir, project_sample_dir):
            if os.path.exists(p):
                extra_includes.extend(["-I", p])
        include_flags = [*include_flags, *extra_includes]

        # Run Clang static analyzer to dump CFG.
        # Set clang_args depending on file extension (CUDA or C++)
        clang_args = []
        if tmp_path.endswith(".cu"):
            clang_args = [
                "clang++",
                "-x", "cuda",                             # parse as CUDA source
                "--cuda-path=/usr/local/cuda",            # point to the CUDA SDK you installed
                "--no-cuda-version-check",                # avoid strict version compatibility checks
                "--cuda-gpu-arch=sm_70",                  # set appropriate GPU arch (see note below)
                "-std=c++17",                             # Use C++17 standard for host code 
                "-DFLOAT_BITS=64",                        # ensure consistent floating-point behavior
                #"-D__forceinline__=inline", 
                #"-D__noinline__=__attribute__((noinline))",
                #"-D__forceinline__=__attribute__((always_inline))"
                "-fopenmp",
                "-march=native",                          # Enable all CPU instruction sets available locally
                #"-fno-inline-functions",
                #"-fno-inline-functions-called-once",
                "-fsyntax-only",                          # syntax-only (we only need CFG)
                "-O0",
                "-g",
                "-Xclang", "-analyze",
                "-Xclang", "-analyzer-checker=debug.DumpCFG",
                "-I", "/usr/local/cuda/include",          # make sure CUDA include dir is visible
                *include_flags,                           # system include flags from your helper
                tmp_path,
            ]
        else:
            clang_args = [
                "clang++",       # Invoke Clang C++ compiler.
                "-std=c++17",    # Use C++17 standard.
                "-DFLOAT_BITS=64",
                "-fopenmp",      # Support OpenMP pragmas.
                "-march=native", # Enable all CPU instruction sets available locally.
                "-fsyntax-only", # Only check syntax, do not compile.
                "-O0",           # Disable optimizations for a clearer CFG.
                "-g",            # Include debug info.
                "-nostdinc",     # Do not use default system includes.
                "-Xclang", "-analyze",
                "-Xclang", "-analyzer-checker=debug.DumpCFG",
                *include_flags,
                tmp_path,
            ]
        process = subprocess.run(
            clang_args,
            capture_output=True,
            text=True,
        )

        output = process.stdout + "\n" + process.stderr
        #print(output)  # Debugging: inspect Clang CFG dump.
        '''
        The Core idea: is to chain function building -> node building -> successor building
        At Any point one has a state, where the program is in a function with an isolated CFG,
        and the current node is the last node added to that CFG.
        Only a function invocation resets the current node.
        Likewise only a node_match to hat node edges are being added.
        This allows us to build CFGs for each function independently,
        and then compute cyclomatic complexity for each function separately.
        '''
        function_cfgs = build_cfg_from_dump(output)
        # Compute total cyclomatic complexity.
        total_complexity = 0
        for func_name, func_cfg in function_cfgs.items():
            edges = func_cfg.number_of_edges()
            nodes = func_cfg.number_of_nodes()
            cyclomatic_complexity = edges - nodes + 2 * 1  # P = 1 per function.
            print(
                f"Function {func_name}: E={edges}, N={nodes}, P=1, CC={cyclomatic_complexity}"
            )
            total_complexity += cyclomatic_complexity
        return total_complexity
    finally:
        # Ensure temporary file is removed even on error.
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            
def remove_comments_with_clang(code: str, filename: str = "tmp.cpp") -> str:
    """Remove comments using Clang's tokenization."""
    # Configure libclang first
    if sys.platform.startswith("linux"):
        libclang_path = "/usr/lib/llvm-18/lib/libclang.so"
    elif sys.platform.startswith("darwin"):
        libclang_path = "/usr/local/opt/llvm/lib/libclang.dylib"
    elif sys.platform.startswith("win32"):
        libclang_path = r"C:\Program Files\LLVM\bin\libclang.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")
    clang.cindex.Config.set_library_file(libclang_path)
    # Then the rest of your function
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    index = clang.cindex.Index.create()
    tu = index.parse(filename, args=["-std=c++17"])
    tokens = tu.get_tokens(extent=tu.cursor.extent)
    return "".join(t.spelling for t in tokens if t.kind != clang.cindex.TokenKind.COMMENT)