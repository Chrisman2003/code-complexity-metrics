import networkx as nx
import subprocess
import re
import tempfile
import os
import clang.cindex
import sys

# -----------------------------------------------------------------------------
# Cyclomatic Complexity Analysis for C++ Code
# -----------------------------------------------------------------------------
# This module computes the McCabe cyclomatic complexity (V(G) = E − N + 2P)
# for C++ code using Clang's static analyzer for precise CFG extraction,
# as well as a simpler heuristic method based on control-flow keywords.
# -----------------------------------------------------------------------------
    
def get_clang_include_flags() -> list[str]:
    """Construct Clang-compatible include flags for system headers.

    This function builds a list of `-isystem` flags to ensure Clang can find
    all necessary headers when analyzing C++ or OpenCL code. It merges:

    1. Clang's own builtin headers (via `clang++ -print-resource-dir`).
    2. GCC's standard library include paths (via `g++ -E -x c++ - -v`).
    3. Fallback system include paths (`/usr/include`, `/usr/local/include`).

    Returns:
        list[str]: A list of `-isystem <path>` flags to pass to Clang.
    """
    # Step 1: Add Clang's builtin resource headers.
    # These headers include stddef.h, stdint.h, and OpenCL-specific files
    # (e.g., opencl-c.h) that GCC's libstdc++ does not provide.
    clang_resource_dir = subprocess.run(
        ["clang++", "-print-resource-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip() # clang_resource_dir holds actual string result of clang++ -print-resource-dir.
    include_flags = ["-isystem", os.path.join(clang_resource_dir, "include")]
    # Step 2: Add GCC system include directories.
    # Extract from verbose output of `g++` preprocessing.
    # This ensures that downstream Clang analysis (in compute_cyclomatic) can locate all
    # standard headers on any machine/architecture
    process = subprocess.run(
        ["g++", # Invoke GCC C++ compiler
         "-E",  # preprocess only
         "-x",  # specify language
         "c++", # treat input as C++
         "-",   # read from stdin
         "-v"], # verbose
        input="",
        text=True,
        capture_output=True,
    )

    in_block = False
    for line in process.stderr.splitlines():
        if "#include <...> search starts here:" in line:
            in_block = True
            continue
        if "End of search list." in line:
            break
        if in_block:
            path = line.strip()
            include_flags.extend(["-isystem", path])
    # Step 3: Add fallback system includes if present.
    # <iostream> lives here on some systems
    # /usr/lib
    for path in ("/usr/include", "/usr/local/include"):
        if os.path.exists(path):
            include_flags.extend(["-isystem", path])
    return include_flags

def build_cfg_from_dump(output: str) -> dict[str, nx.DiGraph]:
    """Build control-flow graphs (CFGs) for each function from Clang's CFG dump.

    Parses the output of `clang++ -Xclang -analyze -Xclang -analyzer-checker=debug.DumpCFG`
    to construct a mapping from function names to their corresponding
    `networkx.DiGraph` representations of the control-flow graph.

    Args:
        output (str): The raw textual output from Clang's CFG dump.

    Returns:
        dict[str, nx.DiGraph]: A dictionary where keys are function names and values
        are directed graphs (`nx.DiGraph`) representing the function's control-flow.
    """
    node_pattern = re.compile(r'\[B(\d+)(?: \([A-Z]+\))?\]')
    succ_pattern = re.compile(r'Succs \((\d+)\): (.+)')
    func_start_pattern = re.compile(r'^\s*(\w[\w\s:*&<>]*)\s+(\w+)\(.*\)\s*$')
    
    function_cfgs = {}
    current_function = None
    current_node = None
    cfg = nx.DiGraph()
    for line in output.splitlines():
        func_match = func_start_pattern.match(line)
        node_match = node_pattern.search(line)
        succ_match = succ_pattern.search(line)
        if func_match:
            # Start a new function CFG.
            current_function = func_match.group(2)
            cfg = nx.DiGraph()
            function_cfgs[current_function] = cfg
            current_node = None
            continue
        if current_function is None:
            # Skip lines outside of function CFGs.
            continue
        if node_match:
            # Found a new CFG node.
            current_node = int(node_match.group(1))
            cfg.add_node(current_node)
        if succ_match and current_node is not None:
            # Add edges from the current node to its successors.
            successors = [int(s[1:]) for s in succ_match.group(2).split()]
            for succ in successors:
                cfg.add_edge(current_node, succ)
    return function_cfgs

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
        clang_args = [
            "clang++",       # Invoke Clang C++ compiler.
            "-std=c++17",    # Use C++17 standard.
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

def remove_cpp_comments(code: str) -> str:
    # Remove all block comments first
    code_no_block = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove single line comments
    code_no_comments = re.sub(r'//.*', '', code_no_block)
    return code_no_comments

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

def basic_compute_cyclomatic(code: str) -> int:
    """
    Compute a simplified cyclomatic complexity estimate using heuristics.
    
    Accepts either the source text via `code` or a path via `filename`.
    If `filename` is provided it will be read (UTF-8, ignore errors) and parsed.

    Args:
        code: Source code as a string (optional if filename provided).
        filename: Path to source file to load (optional).

    Returns:
        int: Heuristic cyclomatic complexity value.
    """
    #if filename:
    #    try:
    #        with open(filename, "r", encoding="utf-8", errors="ignore") as fh:
    #            code = fh.read()
    #    except Exception:
    #        # If file cannot be read, fall back to empty source so result is 1
    #        code = ""
    #if code is None:
    #    code = ""
    code = remove_cpp_comments(code)
    #code = remove_comments_with_clang(code, filename="tmp.cpp") # Alternative using clang  
    # Keywords and operators that increase cyclomatic complexity
    control_keywords = ['if', 'for', 'while', 'case', 'default', 'catch', 'else', 'do', 'goto']
    # Switch itself doesn't add to cyclomatic complexity, only case labels do.
    logical_operators = ['&&', '||', '?', 'and', 'or']
    count = 0
    for line in code.splitlines():
        stripped = line.strip() # Leading and Trailing whitespaces removed
        for keyword in control_keywords:
            if re.search(rf'\b{keyword}\b', stripped): # Only match with word boundaries
                # Edge Cases: 
                # -> label: 
                # -> if (x > 0) 
                # -> } if (x > 0)
                count += 1
        for op in logical_operators:
            count += stripped.count(op)
    return count + 1  # +1 for the default path

'''
Edge Cases for Commenting:
1) 
std::string s = "This is not a // comment";
char c = '/';
std::string t = "/* not a comment */";
2)
/* Outer comment
   /* Inner comment */
   End of outer */
3) 
#define STR(x) "/* " #x " */"
'''