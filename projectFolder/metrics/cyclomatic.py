import networkx as nx
import subprocess
import re
import tempfile

# Mccabe Cyclomatic Complexity Calculation for C++ Code
# Formula Spefication: V(G)=E−N+2P

def get_gcc_include_paths():
    """Ask g++ for its include search paths for C++."""
    process = subprocess.run(
        ["g++", "-E", "-x", "c++", "-", "-v"],
        input="",
        text=True,
        capture_output=True
    )
    includes = []
    in_block = False
    for line in process.stderr.splitlines():
        if "#include <...> search starts here:" in line:
            in_block = True
            continue
        if "End of search list." in line:
            break
        if in_block:
            includes.append(line.strip())
    return includes

# Real Mccabe Cyclomatic Complexity Calculation
def compute_cyclomatic(code: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    # Get platform-dependent GCC include paths
    include_paths = get_gcc_include_paths()
    include_flags = []
    for path in include_paths:
        include_flags.extend(["-isystem", path])

    process = subprocess.run(
        [
            "clang++",
            "-std=c++17",
            "-fsyntax-only",
            "-Xclang", "-analyze",
            "-Xclang", "-analyzer-checker=debug.DumpCFG",
            *include_flags,
            tmp_path,
        ],
        capture_output=True,
        text=True
    )
    output = process.stdout + "\n" + process.stderr
    # The stdout may actually be empty because the analyzer prints the CFG to stderr, not stdout. In your code you only read stdout.
    print("Clang CFG Dump:\n", output)  # Debug: see what Clang outputs
    cfg = nx.DiGraph()
    # Parse nodes and successors from Clang CFG dump
    node_pattern = re.compile(r'\[B(\d+)\]') # Matches Blocknodes with corresponding IDs
    succ_pattern = re.compile(r'Succs \((\d+)\): (.+)') # Matches no. successors as well as corresponding Node IDs
    current_node = None
    for line in output.splitlines():
        node_match = node_pattern.search(line)
        succ_match = succ_pattern.search(line)
        
        if node_match:
            current_node = int(node_match.group(1))
            cfg.add_node(current_node)
        
        if succ_match and current_node is not None:
        # Remove 'B' prefix and convert to int
            successors = [int(s[1:]) for s in succ_match.group(2).split()]
            for succ in successors:
                cfg.add_edge(current_node, succ)
                
    # Count edges (E), nodes (N) and connected components (P)
    edges = cfg.number_of_edges()
    nodes = cfg.number_of_nodes()
    components = nx.number_connected_components(cfg.to_undirected())
    # Calculate Cyclomatic Complexity
    cyclomatic_complexity = edges - nodes + 2 * components
    print(f"Cyclomatic Complexity: {cyclomatic_complexity} (E={edges}, N={nodes}, P={components})")
    # Return the cyclomatic complexity value
    return cyclomatic_complexity

# Simplified Heuristic
def basic_compute_cyclomatic(code: str) -> int:
    # C++ control flow keywords and logical operators
    control_keywords = [
        'if', 'for', 'while', 'case', 'catch', 'switch', 'else', 'do', 'goto'
    ]
    logical_operators = ['&&', '||', '?', 'and', 'or']  # 'and', 'or' for alternative tokens
    count = 0 
    for line in code.splitlines():
        stripped = line.strip()
        # Count control keywords at the start of the line (simple heuristic)
        for keyword in control_keywords:
            if stripped.startswith(keyword):
                count += 1
        # Count logical operators anywhere in the line
        for op in logical_operators:
            count += stripped.count(op)
    return count + 1  # +1 for


