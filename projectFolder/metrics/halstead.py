import re
import math
from .KEYWORDS import cpp_non_operands, cuda_non_operands, kokkos_non_operands, opencl_non_operands

def halstead_metrics_parametrized(code: str, operator_pattern: str, operand_pattern: str, subtracting_set: set):
    """
    Parametrized Halstead metrics calculation.
    """
    operators = re.findall(operator_pattern, code)
    operands = re.findall(operand_pattern, code)
    operands = [op for op in operands if op not in subtracting_set]
    n1 = len(set(operators))
    n2 = len(set(operands))
    N1 = len(operators)
    N2 = len(operands)
    return {
        'n1': n1,
        'n2': n2,
        'N1': N1,
        'N2': N2
    }
        
#operator_pattern = (
#    r'\b(?:if|else|switch|case|for|while|do|break|continue|return|goto|try|catch|throw|new|delete|sizeof|typeid|dynamic_cast|static_cast|reinterpret_cast|const_cast|and|or|not|xor|bitand|bitor|compl)\b'
#    r'|==|!=|<=|>=|->|->\*|<<=|>>=|<<|>>|&&|\|\||\+\+|--|::|\.|->|[-+*/%=&|^~!]=?|[(){}\[\],;:]'
#)
''' merged_nonoperators = core_non_operands | additional_non_operands # Merged with C++ keywords
    escaped_operators = sorted(
        (re.escape(op) for op in merged_nonoperators),
        key=len,
        reverse=True
    )    
    operator_pattern = r'|'.join(escaped_operators)
    operand_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b'
    subtracting_set = merged_nonoperators
    return operator_pattern, operand_pattern, subtracting_set
'''
# Assuming any non-operand is an operator
# Bug, must match operators in a specific order, so as to not match too early on suboperands
# For example matching on = in an "==" expression prematurely
# would lead to incorrect results.
# Sorted keywords for deterministric matching
# and to avoid matching substrings of longer operators
# Python set is unordered, meaning it doesn’t store elements in the order you declare them.

def compute_sets(core_non_operands: set, additional_non_operands: set) -> set:
# Merge C++ core non-operands with extension-specific ones
    merged_nonoperators = core_non_operands | additional_non_operands
    # Split into keyword-like (alphanumeric & underscores) and symbolic operators
    keyword_ops = {op for op in merged_nonoperators if re.match(r'^[A-Za-z_]\w*$', op)}
    symbol_ops = merged_nonoperators - keyword_ops
    # Escape for regex
    escaped_keywords = [r'\b' + re.escape(op) + r'\b' for op in sorted(keyword_ops)]
    escaped_symbols  = [re.escape(op) for op in sorted(symbol_ops, key=len, reverse=True)]
    # Combine into one pattern: keywords first, then symbols
    operator_pattern = r'|'.join(escaped_keywords + escaped_symbols)
    # Otherwise if might match diff → wrong operand/metric counts
    # >> might be partially matched as > → wrong operator counts
    # Operand pattern: identifiers & numbers
    operand_pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b'
    subtracting_set = merged_nonoperators
    return operator_pattern, operand_pattern, subtracting_set

def halstead_metrics_cpp(code: str):   
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, set())
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)

def halstead_metrics_cuda(code: str):
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, cuda_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)

def halstead_metrics_kokkos(code: str):
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, kokkos_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)

def halstead_metrics_opencl(code: str):
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, opencl_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)

'''
Formulas for Halstead Metrics:
- n = Vocabulary = n1 + n2
- N = Size = N1 + N2
- V = Volume = N*log2(n)
- D = Difficulty = (n1/2) * (N2/n2)
- E = Effort = D * V
- T = Time = E / 18
--> 18 is the arbitrary default value for the stroud number
'''    

def vocabulary(metrics: dict) -> int:
    return metrics['n1'] + metrics['n2']

def size(metrics: dict) -> int:
    return metrics['N1'] + metrics['N2']

def volume(metrics: dict) -> float:
    return size(metrics) * math.log2(vocabulary(metrics))

def difficulty(metrics: dict) -> float: 
    if metrics['n2'] == 0:
        return 0.0
    return (metrics['n1'] / 2) * (metrics['N2'] / metrics['n2'])

def effort(metrics: dict) -> float:
    return difficulty(metrics) * volume(metrics)

def time(metrics: dict) -> float:
    return effort(metrics) / 18
