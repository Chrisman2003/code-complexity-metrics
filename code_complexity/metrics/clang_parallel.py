import networkx as nx
import subprocess
import tempfile
import os
import clang.cindex
import sys
from clang import cindex
from clang.cindex import CursorKind
from code_complexity.metrics.utils import *

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


# Flow-breaking AST kinds (common ones)
FLOW_KINDS = {
    CursorKind.IF_STMT,
    CursorKind.FOR_STMT,
    CursorKind.WHILE_STMT,
    CursorKind.DO_STMT,
    CursorKind.SWITCH_STMT,
    CursorKind.CXX_CATCH_STMT,  # catch
    CursorKind.CONDITIONAL_OPERATOR,  # ternary ?:, sometimes separate
    # CursorKind.CASE_STMT: # case labels may be handled inside switch
}

# Jump-like statements that break linear flow (count them but they don't open nesting)
JUMP_KINDS = {
    CursorKind.GOTO_STMT,
    CursorKind.BREAK_STMT,
    CursorKind.CONTINUE_STMT,
    # labelled breaks/gotos: same handling
}

# Function-like declarations to find function bodies
FUNCTION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.LAMBDA_EXPR,  # if you want to score lambdas
}


def _count_logical_ops_in_expr(node):
    """Count logical operators '&&' and '||' in an AST expression subtree.

    This is used to add cognitive complexity for expressions with multiple
    logical paths.

    Args:
        node (clang.cindex.Cursor): AST node representing an expression.

    Returns:
        int: Number of logical operators in the subtree.
    """
    count = 0
    try:
        tokens = list(node.get_tokens())
    except Exception:
        return 0
    for t in tokens:
        if t.spelling in ("&&", "||"):
            count += 1
    return count


def compute_cognitive_for_function(func_cursor):
    """Compute Cognitive Complexity for a single function AST.

    Traverses the function body and calculates cognitive complexity based
    on nesting, flow-breaking statements, and logical operators.

    Args:
        func_cursor (clang.cindex.Cursor): Cursor pointing to the function or method declaration.

    Returns:
        int: Cognitive complexity score for the function.
    """
    score = 0
    nesting = 0

    def visit(node):
        """Recursive AST traversal helper to compute complexity.

        Args:
            node (clang.cindex.Cursor): AST node being visited.
        """
        nonlocal score, nesting

        # Immediate flow-breakers that increase nesting
        if node.kind in FLOW_KINDS:
            score += 1 + nesting
            nesting += 1
            for c in node.get_children():
                visit(c)
            nesting -= 1
            return

        # Jumps (break/continue/goto) break linear flow but do not open nesting
        if node.kind in JUMP_KINDS:
            score += 1 + nesting
            for c in node.get_children():
                visit(c)
            return

        # Count logical operators inside expressions
        if node.kind == CursorKind.BINARY_OPERATOR or node.kind == CursorKind.CONDITIONAL_OPERATOR:
            ops = _count_logical_ops_in_expr(node)
            if ops:
                score += ops * (1 + nesting)

        # Default traversal (no nesting change)
        for c in node.get_children():
            visit(c)

    # Traverse function body (usually a CompoundStmt)
    for child in func_cursor.get_children():
        if child.kind.is_statement() or child.kind == CursorKind.COMPOUND_STMT:
            visit(child)
            break

    return score


def compute_cognitive_complexity_file(filename):
    """Compute Cognitive Complexity for all functions in a C++ file.

    Parses the file using clang AST and calculates complexity per function.

    Args:
        filename (str): Path to the C++ source file.

    Returns:
        int: Total cognitive complexity across all functions in the file.
    """
    index = cindex.Index.create()
    args = ["-std=c++17"]
    tu = index.parse(filename, args=args)

    results = {}
    total_score = 0

    # Walk top-level cursors to find function/method definitions
    for cursor in tu.cursor.walk_preorder():
        if cursor.kind in FUNCTION_KINDS:
            has_body = any(c.kind == CursorKind.COMPOUND_STMT for c in cursor.get_children())
            if not has_body:
                continue
            name = cursor.displayname or cursor.spelling or "<anon>"
            score = compute_cognitive_for_function(cursor)
            results[name] = score
            total_score += score

    return total_score