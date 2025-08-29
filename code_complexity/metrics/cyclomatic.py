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
    ).stdout.strip()
    include_flags = ["-isystem", os.path.join(clang_resource_dir, "include")]

    # Step 2: Add GCC system include directories.
    # Extract from verbose output of `g++` preprocessing.
    # This ensures that downstream Clang analysis (in compute_cyclomatic) can locate all 
    # standard headers on any machine/architecture
    process = subprocess.run(
        ["g++", "-E", "-x", "c++", "-", "-v"],
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
    for path in ("/usr/include", "/usr/local/include"):
        if os.path.exists(path):
            include_flags.extend(["-isystem", path])

    return include_flags


def compute_cyclomatic(code: str) -> int:
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

    Returns:
        int: Total cyclomatic complexity across all functions.
    """

    # Write code to a temporary file for Clang analysis.
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    # Collect all include flags.
    include_flags = get_clang_include_flags()

    # Run Clang static analyzer to dump CFG.
    process = subprocess.run(
        [
            "clang++",
            "-std=c++17",
            "-march=native",  # Enable all CPU instruction sets available locally.
            "-fsyntax-only",
            "-O0",            # Disable optimizations for a clearer CFG.
            "-g",             # Include debug info.
            "-nostdinc",      # Do not use default system includes.
            "-Xclang", "-analyze",
            "-Xclang", "-analyzer-checker=debug.DumpCFG",
            *include_flags,
            tmp_path,
        ],
        capture_output=True,
        text=True,
    )

    output = process.stdout + "\n" + process.stderr
    print(output)  # Debugging: inspect Clang CFG dump.

    # Regex patterns to capture nodes, successors, and function signatures.
    node_pattern = re.compile(r'\[B(\d+)(?: \([A-Z]+\))?\]')
    succ_pattern = re.compile(r'Succs \((\d+)\): (.+)')
    func_start_pattern = re.compile(r'^\s*(\w[\w\s:*&<>]*)\s+(\w+)\(.*\)\s*$')

    function_cfgs: dict[str, nx.DiGraph] = {}
    cfg = nx.DiGraph()
    current_function = None
    current_node = None

    '''
    The Core idea: is to chain function building -> node building -> successor building
    At Any point one has a state, where the program is in a function with an isolated CFG,
    and the current node is the last node added to that CFG.
    Only a function invocation resets the current node.
    Likewise only a node_match to hat node edges are being added.
    This allows us to build CFGs for each function independently,
    and then compute cyclomatic complexity for each function separately.
    '''
    # Parse the analyzer output line by line.
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


def basic_compute_cyclomatic(code: str) -> int:
    """Compute heuristic cyclomatic complexity of C++ code.

    This function estimates cyclomatic complexity without building a CFG.
    It counts occurrences of control-flow keywords (e.g., if, for, while) and
    logical operators (e.g., &&, ||) as branching points.

    Args:
        code (str): A string containing C++ source code.

    Returns:
        int: Heuristic cyclomatic complexity value.
    """

    control_keywords = [
        "if", "for", "while", "case", "catch", "switch", "else", "do", "goto",
    ]
    logical_operators = ["&&", "||", "?", "and", "or"]

    count = 0
    for line in code.splitlines():
        stripped = line.strip()

        # Count branching keywords at the start of a line.
        for keyword in control_keywords:
            if stripped.startswith(keyword):
                count += 1

        # Count logical operators anywhere in the line.
        for op in logical_operators:
            count += stripped.count(op)

    # Add one for the default execution path.
    return count + 1


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

