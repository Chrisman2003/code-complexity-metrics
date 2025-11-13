import networkx as nx
import subprocess
import re
import tempfile
import os
from code_complexity.metrics.utils import *
import logging
import sys

# -------------------------------
# Logging setup
# -------------------------------
plain_logger = logging.getLogger("plain")
plain_handler = logging.StreamHandler(sys.stdout)
plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain_logger.addHandler(plain_handler)
plain_logger.setLevel(logging.INFO)


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
    
    r""" 
    REGEX EXPLANATION
    Matches nodes that are formatted like "[B<number> (OPTIONAL_LABEL)]".
    Example matches: "[B12]", "[B7 (LOOP)]"
    ------------------------
    -  \[          → Match literal '['
    -  B           → Match literal 'B'
    -  (\d+)       → Capture one or more digits (the node number) into group 1
    -  (?: ...)    → Non-capturing group for optional label
    -      \s      → Match a space
    -      \(      → Match literal '('
    -      [A-Z]+  → Match one or more uppercase letters (the label)
    -      \)      → Match literal ')'
    -  ?           → Make the entire non-capturing group optional
    -  \]          → Match literal ']' 
    """
    node_pattern = re.compile(r'\[B(\d+)(?: \([A-Z]+\))?\]')

    r""" 
    REGEX EXPLANATION:
    This regex matches lines that list successor nodes in a format like:
    "Succs (2): 3, 4, 5"
    -------------------------------
    -  Succs       → Match literal string 'Succs'
    -  \s         → Match a space
    -  \(         → Match literal '('
    -  (\d+)      → Capture one or more digits (number of successors) into group 1
    -  \)         → Match literal ')'
    -  :          → Match literal colon ':'
    -  \s         → Match a space
    -  (.+)       → Capture the rest of the line (comma-separated successor node numbers) into group 2 
    """
    succ_pattern = re.compile(r'Succs \((\d+)\): (.+)')
    
    function_cfgs = {}
    current_function = None
    current_node = None
    cfg = nx.DiGraph()
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if "(ENTRY)" in line:
            # The line above is assumed to be the function declaration
            decl_line = lines[i - 1].strip() if i > 0 else ""
            # Extract function name as the word before first '('
            if '(' in decl_line:
                current_function = decl_line.split('(')[0].split()[-1]
            else:
                current_function = f"unknown_{i}"
            cfg = nx.DiGraph()
            function_cfgs[current_function] = cfg
            current_node = None
            continue    
        
        node_match = node_pattern.search(line)
        succ_match = succ_pattern.search(line)
        
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
        clang_args = [
            "clang++",       # Invoke Clang C++ compiler.
            "-std=c++17",    # Use C++17 standard.
            "-fopenmp",      # Support OpenMP pragmas.
            "-march=native", # Enable all CPU instruction sets available locally.
            "-fsyntax-only", # Only check syntax, do not compile.
            "-O0",           # Disable optimizations for a clearer CFG.
            "-g",            # Include debug info.
            "-nostdinc",     # Do not use default system includes.
            "-Xclang", "-analyze", # Enable static analysis.
            "-Xclang", "-analyzer-checker=debug.DumpCFG", # Dump the CFG.
            *include_flags,
            tmp_path,
        ]
        process = subprocess.run( # Run Clang static analyzer to dump CFG.
            clang_args,
            capture_output=True,
            text=True,
        )

        output = process.stdout + "\n" + process.stderr
        
        # Save CFG to a file
        #cfg_file_path = os.path.join(original_dir, "clang_cfg_dump.txt")
        #with open(cfg_file_path, "w", encoding="utf-8") as f:
        #    f.write(output)
        #plain_logger.info(f"Clang CFG dump saved to: {cfg_file_path}")
        plain_logger.info(output)  # Debugging: inspect Clang CFG dump.
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
        total_complexity = 0 # Compute total cyclomatic complexity.
        for func_name, func_cfg in function_cfgs.items():
            edges = func_cfg.number_of_edges()
            nodes = func_cfg.number_of_nodes()
            cyclomatic_complexity = edges - nodes + 2 * 1  # P = 1 per function.
            plain_logger.info(f"Function {func_name}: E={edges}, N={nodes}, P=1, CC={cyclomatic_complexity}")
            total_complexity += cyclomatic_complexity
        return total_complexity
    finally:
        # Ensure temporary file is removed even on error.
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def detect_fallthroughs(code: str) -> int:
    fallthroughs = 0
    switch_blocks = re.findall(r'\bswitch\b\s*\([^)]*\)\s*\{([^}]*)\}', code, re.DOTALL)
    for block in switch_blocks:
        # Split each switch block into case segments
        cases = re.split(r'\bcase\b|\bdefault\b', block)
        # Skip first segment (before first case)
        #print(cases)
        for seg in cases[1:-1]:
            # Only count if segment lacks 'break;'
            if 'break;' not in seg:
                fallthroughs += 1
    return fallthroughs

def basic_compute_cyclomatic(code: str) -> int:
    """
    Compute a simplified cyclomatic complexity estimate using heuristics.
    
    Accepts either the source text via `code` or a path via `filename`.
    If `filename` is provided it will be read (UTF-8, ignore errors) and parsed.

    Args:
        code (str): Source code as a string. The code should be plain text; 
            comments and string literals are removed internally before analysis.
    """
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code = remove_string_literals(code)
    control_keywords = ['if', 'for', 'while', 'case', 'default', 'catch', 'do', 'goto']
    logical_operators = ['&&', '||', '?', 'and', 'or']
    count = 0
    num_functions = 0
    # Count function definitions (ignoring function calls!)
    function_pattern = re.compile(r"""
        (?:template\s*<[^>]+>\s*)*                     # optional template declaration(s), e.g., template<typename T>
        (?:[\w:\<\>\~\*\&\s]+\s+)?                     # optional return type including pointers, refs, const, and spaces
        (~?(?:[A-Za-z_]\w*(?:::\w+)*|operator[^\s(]+)) # function name, scoped names, destructors (~), or operator overloads
        \([^{};]*\)\s*                                 # parameter list in parentheses, flat, no braces or semicolons inside
        (?:const\s*)?(?:noexcept\s*)?                  # optional const and/or noexcept specifiers after parameters
        (?:->\s*[\w:\<\>\s\*&]+)?                      # optional C++11 trailing return type, e.g., -> int
        \s*\{                                          # opening brace of function body to match only definitions
        """, re.VERBOSE | re.DOTALL
    )
    for line in code.splitlines():
        stripped = line.strip() # Leading and Trailing whitespaces removed
        
        # Count Control Keywords
        for keyword in control_keywords:
            if re.search(rf'\b{keyword}\b', stripped): # Only match with word boundaries
                count += 1
                
        # Count Logical Operators
        for op in logical_operators:
            if op in ['and', 'or']:
                count += len(re.findall(rf'\b{op}\b', stripped)) # Alphabetic Operators need word boundaries
            else:
                count += stripped.count(op) # Non-Alphabetic Substrings may be normally counted without boundaries
                # Count functions (definitions only, not calls)
        
        if re.match(function_pattern, stripped):
            # Exclude mis-matched control statements
            if not any(stripped.startswith(k) for k in (control_keywords + ['switch'])):
                #plain_logger.info(f"Function detected: {stripped}") # Show detected function for debugging
                num_functions += 1
    fallthroughs = detect_fallthroughs(code)
    #plain_logger.info("fallthroughs:", fallthroughs)
    return count + num_functions + fallthroughs # +1 for the default path

# Edge Case: Boolean Operators for CFG Method Version

'''
Edge Cases:
    1) Comments (`//`, `/* ... */`) are ignored and do not contribute to complexity.
    2) String literals (e.g., `"x is greater or equal to y"`) are ignored to 
       prevent false positives for keywords like 'or' or 'and'.
    3) The `else` keyword is not counted separately because it does not create 
       an independent path; `else if` is counted via its `if`.
    4) `switch` statements do not add complexity themselves; only `case` and 
       `default` labels are counted.
    5) Logical operators `&&`, `||`, `?` are counted for each occurrence.
    6) Alphabetic logical operators `and`, `or` are counted only when they 
       appear as standalone words, not as substrings of other identifiers.
    7) Multi-line constructs may not be perfectly accounted for in this heuristic.
    8) If `code` is `None`, it is treated as an empty string (CC = 1).
    9) Fall-Through Switch-Case Statements
    10) Clang doesn't produce CFG components on non-instantiated templates
'''

'''
Edge Cases for Commenting:
1) Fall Through branches in switch/case statements
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

'''
Edge Cases: 
-> label: 
-> if (x > 0) 
-> } if (x > 0)
'''

# TODO
'''
Cyclomatic Function Recognition: still not 100%
'''