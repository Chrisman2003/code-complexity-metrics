import networkx as nx
from tree_sitter import get_language, get_parser

# Mccabe Cyclomatic Complexity Calculation for C++ Code
# Formula Spefication: V(G)=E−N+2P

# Real Mccabe Cyclomatic Complexity Calculation
def compute_cyclomatic(code: str) -> int:
    # build control flow graph
    # count edges (E), nodes (N) and connected components (P)
    # 1) Parse to Abstract Syntax Tree (AST)
    parser = get_parser(get_language("cpp"))
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    
    # 2) Build Control Flow Graph (CFG) from AST
    cfg = nx.DiGraph()
    def add_edges(node):
        for child in node.children:
            cfg.add_edge(node.start_byte, child.start_byte)
            add_edges(child)
    add_edges(root_node)
    
    # 3) Count edges (E), nodes (N) and connected components (P)
    edges = cfg.number_of_edges()
    nodes = cfg.number_of_nodes()
    components = nx.number_connected_components(cfg.to_undirected())
    # 4) Calculate Cyclomatic Complexity
    cyclomatic_complexity = edges - nodes + 2 * components
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


