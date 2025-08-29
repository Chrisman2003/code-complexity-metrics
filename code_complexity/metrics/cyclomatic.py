import networkx as nx
import subprocess
import re
import tempfile
import os

# -----------------------------------------------------------------------------
# Cyclomatic Complexity Analysis for C++ Code
# -----------------------------------------------------------------------------
# This module computes the McCabe cyclomatic complexity (V(G) = E − N + 2P)
# for C++ code using Clang's static analyzer for precise CFG extraction,
# as well as a simpler heuristic method based on control-flow keywords.
# -----------------------------------------------------------------------------

def get_clang_include_flags() -> list[str]:
    """
    Construct Clang-compatible -isystem include flags by:
    1. Adding Clang's own builtin headers via `clang++ -print-resource-dir`.
    2. Extracting GCC's standard library include paths using:
         g++ -E -x c++ - -v < /dev/null
    3. Adding /usr/include and /usr/local/include as fallbacks.
    
    Returns:
        List of strings with properly formatted `-isystem <path>` flags.
    """
    # Step 1: Clang resource includes
    # NOTE:
    #   This is critical — Clang ships its own private "resource headers"
    #   (in `<clang>/lib/clang/<version>/include/`) that define builtins
    #   like stddef.h, stdint.h, and also OpenCL-specific headers such as
    #   opencl-c.h. These are not part of GCC’s libstdc++ paths or the
    #   system’s /usr/include tree. Without adding this directory, Clang
    #   will fail to find essential headers when analyzing code that uses
    #   OpenCL or compiler intrinsics.
    clang_resource_dir = subprocess.run(
        ["clang++", "-print-resource-dir"],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()
    include_flags = ["-isystem", os.path.join(clang_resource_dir, "include")]

    # Step 2: GCC system include dirs
    process = subprocess.run(
        ["g++", "-E", "-x", "c++", "-", "-v"],
        input="",
        text=True,
        capture_output=True
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
            
    # Step 3: Common system includes
    for path in ("/usr/include", "/usr/local/include"):
        if os.path.exists(path):
            include_flags.extend(["-isystem", path])        
            
    return include_flags



def compute_cyclomatic(code: str) -> int:
    """
    Compute the McCabe cyclomatic complexity of C++ code using Clang CFG.

    Args:
        code: C++ source code as a string.

    Returns:
        Total cyclomatic complexity across all functions.
    """
    # Write code to temporary file
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    # Get include flags from new merged logic
    include_flags = get_clang_include_flags()

    # Build command just like working CLI
    process = subprocess.run(
        [
            "clang++",
            "-std=c++17",
            "-march=native", # Enable all instruction sets supported by the local machine
            "-fsyntax-only",
            "-O0",          # Disable optimization for clearer CFG
            "-g",           # Optional: include debug info
            "-nostdinc",    # Don't use standard system include paths
            "-Xclang", "-analyze",
            "-Xclang", "-analyzer-checker=debug.DumpCFG", # Dumping Control Flow Graph
            *include_flags,
            tmp_path,
        ],
        capture_output=True,
        text=True
    )

    output = process.stdout + "\n" + process.stderr
    print(output)  # Debug: inspect Clang output

    # Parse nodes and successors from Clang CFG
    # function_cfgs = []  # Store each function's CFG
    # (\d+) → captures the node number (like 9)
    # (?: \([A-Z]+\))? → non-capturing group matching optional space + 
    # parentheses with all caps (ENTRY, EXIT, etc.)
    # ? → makes it optional
    node_pattern = re.compile(r'\[B(\d+)(?: \([A-Z]+\))?\]')
    succ_pattern = re.compile(r'Succs \((\d+)\): (.+)')
    func_start_pattern = re.compile(r'^\s*(\w[\w\s:*&<>]*)\s+(\w+)\(.*\)\s*$')
    # Final Exit Node is by definition reached by some outgoing edge,
    # And hence requires no specialized Regex handling.
    '''Split the Clang output per function
    before building networkx CFGs
    function_cfgs is a Python dictionary
    When you call .items() on a dictionary, 
    it returns an iterator over (key, value) pairs.
    function_cfgs = {
    "foo": cfg1,
    "bar": cfg2
    }
    '''
    function_cfgs = {}
    cfg = nx.DiGraph()
    current_function = None
    current_node = None
    '''
    You want match for function signatures because they usually start the line, 
    possibly with some whitespace.  
    You want search for node labels because they might appear anywhere in the line.
    '''
    
    '''
    The Core idea: is to chain function building -> node building -> successor building
    At Any point one has a state, where the program is in a function with an isolated CFG,
    and the current node is the last node added to that CFG.
    Only a function invocation resets the current node.
    Likewise only a node_match to hat node edges are being added.
    This allows us to build CFGs for each function independently, 
    and then compute cyclomatic complexity for each function separately.
    '''
    
    for line in output.splitlines():
        func_match = func_start_pattern.match(line)
        node_match = node_pattern.search(line)
        succ_match = succ_pattern.search(line)

        if func_match:
            # Start a new function CFG
            current_function = func_match.group(2)
            cfg = nx.DiGraph()  # Reset the CFG for the new function
            function_cfgs[current_function] = cfg
            current_node = None  # Reset current node for the new function
            continue
        if current_function is None:
            continue
        # If it's None, that means the current line of Clang output is not inside
        # any function. So you continue and skip processing that line, 
        # preventing it from being incorrectly added to a function’s CFG.
        ''' The continue keyword in Python is used inside a loop. 
        When Python executes continue, it immediately skips the rest 
        of the current iteration and moves on to the next iteration of the loop.
        '''
        if node_match:
            current_node = int(node_match.group(1))
            cfg.add_node(current_node)
        # Current Node stays the same till Successors are found
        if succ_match and current_node is not None:
            successors = [int(s[1:]) for s in succ_match.group(2).split()]
            for succ in successors:
                cfg.add_edge(current_node, succ)

    # edges = cfg.number_of_edges()
    # nodes = cfg.number_of_nodes()
    # components = nx.number_connected_components(cfg.to_undirected())
    # cyclomatic_complexity = edges - nodes + 2 * components
    # print(cfg.edges())  # Debug: inspect edges in the CFG
    # print(cfg.nodes())  # Debug: inspect nodes in the CFG
    # print(f"Cyclomatic Complexity: {cyclomatic_complexity} (E={edges}, N={nodes}, P={components})")
    total_complexity = 0
    for func_name, func_cfg in function_cfgs.items():
        edges = func_cfg.number_of_edges()
        nodes = func_cfg.number_of_nodes()
        # Each function is an isolated CFG, so P=1
        cyclomatic_complexity = edges - nodes + 2 * 1
        print(f"Function {func_name}: E={edges}, N={nodes}, P=1, CC={cyclomatic_complexity}")
        total_complexity += cyclomatic_complexity
    return total_complexity
    # Summing per iteration is algebraically equivalent to computing the cyclomatic complexity of the entire codebase
    # Along with the isolated mini CFGs for each function


def basic_compute_cyclomatic(code: str) -> int:
    """
    Compute a simplified cyclomatic complexity estimate using heuristics.
    Counts occurrences of control flow keywords and logical operators.
    
    Args:
        code: A string containing the C++ source code.
    
    Returns:
        int: Heuristic cyclomatic complexity value.
    """
    control_keywords = ['if', 'for', 'while', 'case', 'catch', 'switch', 'else', 'do', 'goto']
    logical_operators = ['&&', '||', '?', 'and', 'or']
    count = 0

    for line in code.splitlines():
        stripped = line.strip()
        for keyword in control_keywords:
            if stripped.startswith(keyword):
                count += 1
        for op in logical_operators:
            count += stripped.count(op)

    return count + 1  # +1 for the default path

