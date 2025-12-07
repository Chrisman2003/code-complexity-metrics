import re
from code_complexity.metrics.utils import *

def count_logical_sequences(line: str, nesting: int) -> int:
    """Counts logical operator sequences for Cognitive Complexity.

    This function scans a line of code and detects occurrences of the logical
    operators ``&&`` and ``||``. It increments complexity when a logical
    operator begins a *new* sequence (i.e., it differs from the previous
    operator encountered). Each sequence contributes a base cost of 1 plus an
    additional weight equal to the current nesting level.

    Args:
        line (str): A single line of source code to analyze.
        nesting (int): The current nesting depth that increases the weight of
            each new logical sequence.

    Returns:
        int: The computed logical-sequence contribution to Cognitive Complexity.
    """
    complexity = 0
    prev_op = None
    for match in re.finditer(r'&&|\|\|', line): # Find all logical operators
        op = match.group() # Current operator
        if op != prev_op: # Only count if different from previous -> Sequence Chain Counting
            complexity += 1 + nesting  # one increment per sequence, weighted by nesting
            prev_op = op
    return complexity

def regex_compute_cognitive(code: str) -> int:
    """Compute Cognitive Complexity (SonarQube Specification) score using regex-based analysis.

    This function scans lines of code and detects occurrences of control-flow
    structures, jump statements, logical operators, and nesting changes, using
    the scoring rules described in SonarQube's Cognitive Complexity specification.
    
    Args:
        code (str): A string containing C++ source code.

    Returns:
        int: Cognitive Complexity score for the source code.
    """   
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code = remove_string_literals(code)
    control_flow_total = 0
    jumps_total = 0
    logical_total = 0
    ternary_total = 0 
    
    # Control flow keywords (notice: no plain 'else')
    control_keywords = ['if', 'for', 'while', 'switch', 'catch'] # 'do' and 'if else' implicit
    # switch with branches add 1, 
    logical_ops_alpha = ['and', 'or']  # counted per occurrence

    complexity = 0
    nesting = 0
    found_nesting_keyword = False # NEW
    for line in code.splitlines():
        stripped = line.strip()
        # --- Control keywords ---
        for keyword in control_keywords: # Count control keywords
            matches = re.findall(rf'\b{keyword}\b', stripped) # Word boundaries to avoid false positives
            add = len(matches) * (1 + nesting)
            complexity += add
            control_flow_total += add
            
        # --- Jump statements ---
        if re.search(r'\bgoto\b\s+\w+', stripped):
            add = 1 + nesting
            complexity += add
            jumps_total += add
        if re.search(r'\bbreak\b\s+\w+', stripped) or re.search(r'\bbreak\s+\d+', stripped):
            add = 1 + nesting
            complexity += add
            jumps_total += add
        if re.search(r'\bcontinue\b\s+\w+', stripped) or re.search(r'\bcontinue\s+\d+', stripped):
            add = 1 + nesting
            complexity += add
            jumps_total += add
        # -> Non-Parametrized Constructs aren't being matched
        
        # --- Logical operators ---
        add = count_logical_sequences(stripped, nesting)
        complexity += add
        logical_total += add
        
        # --- Ternary operator ---
        add = stripped.count('?') * (1 + nesting)
        complexity += add
        ternary_total += add
        
        for op in logical_ops_alpha:
            matches = re.findall(rf'\b{op}\b', stripped) # Word boundaries to avoid false positives
            complexity += len(matches) * (1 + nesting)
            
        nesting = max(0, nesting - stripped.count('}'))
        # Increase nesting ONLY if a nesting keyword and a brace are on this line.
        if (stripped != '{' and stripped != ""):
            found_nesting_keyword = False
        for keyword in control_keywords:
            if re.search(rf'\b{keyword}\b', stripped): # Word boundaries to avoid false positives
                found_nesting_keyword = True
                break
        if found_nesting_keyword and '{' in stripped: # No nesting for if (n == 0) return 1.0;
            nesting += stripped.count('{')    
            found_nesting_keyword = False
    plain_logger.debug(
        f"Control Flow = {control_flow_total}\n"
        f"Jumps = {jumps_total}\n"
        f"Logical Ops = {logical_total}\n"
        f"Ternary = {ternary_total}"
    )
    return complexity

''' 
EDGE CASE DOCUMENTATION:
HANDLED:
1) Misalignment of Nesting and Keywords
if (condition) // 'if' is found here, scored with current 'nesting'
{              // Nesting level actually increases here for the block
    //...
}
2) Over-Penalizing Logical Operators: The metric applies the full nesting penalty to logical operators
3) Word Boundaries: Ensure keywords are matched as whole words to avoid false positives
4) Comments and Strings: Ensure that keywords within comments or string literals do not affect complexity
5) Nesting only increases from control keywords (Unlike Nesting Depth purely with '{' and '}' tokens)
6) Cognitive Complexity unlike Cyclomatic Complexity starts at 0 not 1!
7) Null-Coalescing Operators -> Doesn't exist in C++
8) Switch statements: 'cases' don't add complexity, only the switch statement itself [SonarQube specification]
9) Sequences of like-boolean operators increment complexity, not the operators themselves
10) goto, break and continue statements only contribute to complexity when parametrized

NOT HANDLED:
1) Ignoring Non-Brace Scope
    for (int i = 0; i < 10; ++i)
        if (i % 2 == 0) return; // 'if' is nested, but 'nesting' remains 0.            
2) Nesting without '{' and '}' tokens
3) Fundamental increment for recursion (considered a “meta-loop”)
-> Direct recursion → +1 per function | Indirect recursion (cycles) → +1 per function in the cycle.
-> Nesting still applies for any control flow inside the function.
4) There is no structural increment for lambdas, nested methods, and similar features, but such methods 
do increment the nesting level when nested inside other method-like structures:
'''