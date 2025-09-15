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
    """Cognitive Complexity counts extra complexity for every logical operator (&&, ||) in a condition —
       because they make the reader mentally evaluate multiple paths.
       Count '&&' and '||' occurrences inside an expression subtree.
       libclang exposes operators as tokens; hence collect tokens for the subtree and count matches.
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

def compute_cognitive_for_function(func_cursor: str) -> int:
    """Compute Cognitive Complexity score for C++ source code, using an AST parser
    
    (e.g., clang.cindex) to correctly handle macros, lambdas, and templates.

    Args:
        code (str): The entire C++ source file contents as a string.

    Returns:
        int: Cognitive Complexity score for the given code.
    """
    # Placeholder implementation
    score = 0
    nesting = 0
    
    
    def visit(node):
        nonlocal score, nesting

        # Defensive: skip nodes from other files (if you parse a TU with multiple files)
        # if node.location.file and node.location.file.name != func_cursor.location.file.name:
        #     return

        # If the node is an "immediate" flow-breaker that *also* increases nesting:
        if node.kind in FLOW_KINDS:
            # Add base increment + nesting penalty
            score += 1 + nesting

            # Special-case: conditional operator (?:) may have children directly representing branches.
            # Now increase nesting while visiting children so nested constructs get extra weight.
            nesting += 1
            for c in node.get_children():
                visit(c)
            nesting -= 1
            return

        # Jumps (break/continue/goto) count as linear-flow breaks but do not open a nested context
        if node.kind in JUMP_KINDS:
            score += 1 + nesting
            # no nesting push/pop; continue traversal in case of annotations
            for c in node.get_children():
                visit(c)
            return

        # Count logical operators inside expressions (e.g., in if(condition) or binary ops)
        # Approach: detect BinaryOperator or condition expressions and count '&&'/'||' tokens
        if node.kind == CursorKind.BINARY_OPERATOR or node.kind == CursorKind.CONDITIONAL_OPERATOR:
            # count logical operators in token stream for this subtree
            ops = _count_logical_ops_in_expr(node)
            if ops:
                score += ops * (1 + nesting)

        # Function calls are walked (for call-graph later)
        # Default traversal (no nesting change)
        for c in node.get_children():
            visit(c)

    # find function body child: usually a CompoundStmt under the function cursor
    for child in func_cursor.get_children():
        # We'll traverse the body (CompoundStmt) which contains statements
        if child.kind.is_statement() or child.kind == CursorKind.COMPOUND_STMT:
            visit(child)
            break  # usually one body

    return score
    
def compute_cognitive_complexity_file(filename: str) -> int:
    """
    Parse a C++ file and compute per-function cognitive complexity.
    Returns: dict: {function_displayname: score}
    """
    index = cindex.Index.create()
    args = ["-std=c++17"]
    tu = index.parse(filename, args=args)

    results = {}
    total_score = 0
    # walk top-level cursors to find functions/methods
    for cursor in tu.cursor.walk_preorder():
        if cursor.kind in FUNCTION_KINDS:
            # skip declarations without body (extern or prototypes)
            # check for compound statement child
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
