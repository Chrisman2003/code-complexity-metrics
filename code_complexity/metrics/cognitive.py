import re
from clang import cindex
from clang.cindex import CursorKind
from code_complexity.metrics.shared import *
import logging
logger = logging.getLogger("code_complexity")

def count_logical_sequences(line: str, nesting: int) -> int:
    """Count sequences of symbolic logical operators in a line for Cognitive Complexity."""
    complexity = 0
    prev_op = None
    for match in re.finditer(r'&&|\|\|', line): # Find all logical operators
        op = match.group() # Current operator
        if op != prev_op: # Only count if different from previous -> Sequence Chain Counting
            complexity += 1 + nesting  # one increment per sequence, weighted by nesting
            prev_op = op
    return complexity


def regex_compute_cognitive(code: str) -> int:
    """Compute a Cognitive Complexity score using regex-based analysis.
    This function approximates cognitive complexity without using an AST.
    It counts flow-breaking constructs and nesting based on braces.
    Cognitive Complexity is about mental effort to understand, not just paths.
    Note that this is not a simplified implementation, unlike cyclomatic complexity 
    no control flow graph is needed. An AST-based approach might in some cases, however, 
    be more accurate.
    
    Args:
        code (str): C++ source code as a string.

    Returns:
        int: Cognitive Complexity score for the source code.
    """   
    code = remove_cpp_comments(code)
    code = remove_string_literals(code)
    # Control flow keywords (notice: no plain 'else')
    control_keywords = ['if', 'for', 'while', 'switch', 'catch', 'do'] # switch with branches add 1
    #logical_operators = ['&&', '||', '?', 'and', 'or']
    logical_ops_alpha = ['and', 'or']  # counted per occurrence

    complexity = 0
    nesting = 0
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        
        # --- Control keywords ---
        for keyword in control_keywords: # Count control keywords
            matches = re.findall(rf'\b{keyword}\b', stripped) # Word boundaries to avoid false positives
            complexity += len(matches) * (1 + nesting)
            
        # --- Jump statements ---
        if re.search(r'\bgoto\b\s+\w+', stripped):
            complexity += 1 + nesting
        if re.search(r'\bbreak\b\s+\w+', stripped) or re.search(r'\bbreak\s+\d+', stripped):
            complexity += 1 + nesting
        if re.search(r'\bcontinue\b\s+\w+', stripped) or re.search(r'\bcontinue\s+\d+', stripped):
            complexity += 1 + nesting
        # -> Non-Parametrized Constructs aren't being matched
        
        # --- Logical operators ---
        complexity += count_logical_sequences(stripped, nesting)
        complexity += stripped.count('?') * (1 + nesting)  # ':' is ignored
        for op in logical_ops_alpha:
            matches = re.findall(rf'\b{op}\b', stripped) # Word boundaries to avoid false positives
            complexity += len(matches) * (1 + nesting)
            
        nesting = max(0, nesting - stripped.count('}'))
        # Increase nesting ONLY if a nesting keyword and a brace are on this line.
        found_nesting_keyword = False
        for keyword in control_keywords:
            if re.search(rf'\b{keyword}\b', stripped): # Word boundaries to avoid false positives
                found_nesting_keyword = True
                break
        if '?' in stripped:
             found_nesting_keyword = True
        if found_nesting_keyword and '{' in stripped:   
            nesting += stripped.count('{')    
    return complexity

'''
Edge Cases:
    [x] 1) Misalignment of Nesting and Keywords
        if (condition) // 'if' is found here, scored with current 'nesting'
        {              // Nesting level actually increases here for the block
            //...
        }
        -> Only Brace encompassed blocks is penalized with corresponding nesting
    [] 2) Ignoring Non-Brace Scope
        for (int i = 0; i < 10; ++i)
            if (i % 2 == 0) return; // 'if' is nested, but 'nesting' remains 0.
    [x] 3) Over-Penalizing Logical Operators: The metric applies the full nesting penalty to logical operators
    [x] 4) Word Boundaries: Ensure keywords are matched as whole words to avoid false positives.
    [x] 5) Comments and Strings: Ensure that keywords within comments or string literals do not affect complexity.
    [x] 6) Nesting Depth overcounting with function braces 
            -> Nesting only increases when a control keyword and a brace are on the same line.
    [x] 7) Cognitive Complexity unlike Cyclomatic Complexity starts at 0 not 1!
    [] 8) Null-Coalescing Operators
    [x] 9) Switch statements: 'cases' don't add complexity, only the switch statement itself [SonarQube specification]
    [x] 10) Sequences of like-boolean operators increment complexity, not the operators themselves
    [] 11) Fundamental increment for recursion (considered a “meta-loop”)
        -> Direct recursion → +1 per function | Indirect recursion (cycles) → +1 per function in the cycle.
        -> Nesting still applies for any control flow inside the function.
    [x] 12) goto, break and continue statements only contribute to complexity when parametrized
    [] 13) There is no structural increment for lambdas, nested methods, and similar features, but such methods 
    do increment the nesting level when nested inside other method-like structures:
'''

'''
Example that might break nesting [I might have solved it now]:
if (x > 0)
{ // Nesting increases here, not on the if line
    // 
}
The current implementation will score the if on the wrong nesting level if the { is on the next line.
'''