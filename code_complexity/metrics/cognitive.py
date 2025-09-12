import re

def compute_cognitive_complexity(code: str) -> int:
    """Compute a first-version Cognitive Complexity score for C++ source code.

    This function implements the three main rules of Cognitive Complexity
    described by G. Ann Campbell (SonarSource):

    1. Ignore structures that make code more readable (e.g., method declarations).
    2. Increment for each break in the linear flow of the code.
    3. Increment for nesting levels of flow-breaking structures.

    This implementation uses a simple regex-based approach with brace counting
    to track nesting. For production-grade analysis, use an AST parser (e.g.,
    clang.cindex) to correctly handle macros, lambdas, and templates.

    Args:
        code (str): The entire C++ source file contents as a string.

    Returns:
        int: Cognitive Complexity score for the given code.
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
