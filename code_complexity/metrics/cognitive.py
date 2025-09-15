import re
from clang import cindex
from clang.cindex import CursorKind
import collections

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


def basic_compute_cognitive(code: str) -> int:
    """Compute a basic Cognitive Complexity score using regex-based analysis.
    This simpler function approximates cognitive complexity without using an AST.
    It counts flow-breaking constructs and nesting based on braces.

    Args:
        code (str): C++ source code as a string.

    Returns:
        int: Cognitive Complexity score for the source code.
    """
    # Flow-breaking constructs (as defined in Cognitive Complexity rules)
    control_keywords = [
        r'\bif\b', r'\belse if\b', r'\belse\b',
        r'\bfor\b', r'\bwhile\b', r'\bswitch\b',
        r'\bcatch\b', r'\bgoto\b', r'\bcontinue\b', r'\bbreak\b'
    ]
    logical_operators = [r'&&', r'\|\|', r'\?']  # Ternary counted too

    complexity = 0
    nesting = 0

    for line in code.splitlines():
        stripped = line.strip()

        # Track nesting with braces
        # Count '{' after processing constructs, because opening a block increases nesting *for following lines*
        open_braces = stripped.count('{')
        close_braces = stripped.count('}')
        nesting -= close_braces
        if nesting < 0:
            nesting = 0  # avoid negative nesting

        # Increment complexity for each control-flow keyword
        for keyword in control_keywords:
            matches = re.findall(keyword, stripped)
            for _ in matches:
                complexity += 1 + nesting  # +1 per nesting level

        # Increment for each logical operator (adds decision points)
        for op in logical_operators:
            matches = re.findall(op, stripped)
            for _ in matches:
                complexity += 1 + nesting

        # Now update nesting after counting this line
        nesting += open_braces

    return complexity
