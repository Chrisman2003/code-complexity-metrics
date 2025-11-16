import os
import ast
import pygraphviz as pgv

BASE_DIR = "code_complexity"

def find_python_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                yield os.path.join(root, file)

def get_imports(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

G = pgv.AGraph(strict=False, directed=True)

# Add nodes for all files
files = list(find_python_files(BASE_DIR))
for file in files:
    node_name = os.path.relpath(file, BASE_DIR)
    G.add_node(node_name)

# Add edges for imports
for file in files:
    node_name = os.path.relpath(file, BASE_DIR)
    imports = get_imports(file)
    for imp in imports:
        # Only link to files inside the repo
        target_files = [f for f in files if os.path.splitext(os.path.basename(f))[0] == imp]
        for target in target_files:
            target_node = os.path.relpath(target, BASE_DIR)
            G.add_edge(node_name, target_node)

# Layout and save
G.layout(prog="dot")  # or "neato", "fdp"
G.draw("flow_graph.png")
print("Flow graph saved as flow_graph.png")
