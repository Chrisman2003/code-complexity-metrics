# -----------------------------------------------------------------------------
# Halstead Complexity computation utilities for C++ and GPU-extended code
# -----------------------------------------------------------------------------
# Includes:
# - Parametrized token extraction for operators and operands via regex
# - Support for multiple languages and parallel computing frameworks:
#   C++, CUDA, OpenCL, Kokkos, OpenMP, OpenACC, Metal, WebGPU, Boost, OpenGL/Vulkan
# - Automatic handling of framework-specific keywords, functions, and symbols
# - Derived Halstead metrics: vocabulary, program length, volume, difficulty, effort, and time
#
# Note:
# This module provides pattern-based tokenization rather than whitespace-based parsing to correctly handle 
# C++ syntax and its GPU extensions (e.g. "a+b" and "a + b"). Function definitions, calls, literals, and 
# framework-specific constructs are distinguished for precise metric computation. Subtraction sets ensure 
# deterministic operand counting by excluding non-operands. Regex-based extraction ensures robustness across
# code formatting and multi-line constructs.
# -----------------------------------------------------------------------------
import re
import math
from code_complexity.metrics.language_tokens import *
from code_complexity.metrics.utils import *

def halstead_metrics_parametrized(code: str, operator_pattern: str, operand_pattern: str, subtracting_set: set, 
                                   code_str: str, lang: str, autolist: list[str] = []):
    """Compute Halstead metrics for given code with parametrized operator and operand patterns.

    Args:
        code (str): The source code to analyze.
        operator_pattern (str): Regex pattern to match operators.
        operand_pattern (str): Regex pattern to match operands.
        subtracting_set (set): Set of tokens to exclude from operands.

    Returns:
        dict: Dictionary containing Halstead counts:
            - 'n1': Number of distinct operators
            - 'n2': Number of distinct operands
            - 'N1': Total number of operators
            - 'N2': Total number of operands
    """
    # Extract operators and operands using regex
    operators = re.findall(operator_pattern, code)
    operands = re.findall(operand_pattern, code)
    # Only extend with GPU patterns if lang specifies a GPU extension
    dynamic_nonoperands = set()
    patterns_to_apply = []
    if lang == "merged":
        for pat_list in pattern_rules.values():
            patterns_to_apply.extend(pat_list)
    elif lang == "auto" or lang == "cpp":
        for l in autolist:
            if l in pattern_rules:
                patterns_to_apply.extend(pattern_rules[l])
    elif lang in pattern_rules:
        patterns_to_apply = pattern_rules[lang]
    # Result: a list of regex patterns to apply
    # For all regex elements in pattern_rules are used    
    
    for p_elem in patterns_to_apply:
        matches = re.findall(p_elem, code) # find all matches for this pattern
        if lang == "cpp":
            # Subtract Framework functions from Operators in C++
            operators = [op for op in operators if op not in matches]
            dynamic_nonoperands.update(matches) # add to dynamic non-operands
        else:
            operators.extend(matches) # add to operators
            dynamic_nonoperands.update(matches) # add to dynamic non-operands
    
    # 1) Adjust Operands
    func_def_pattern = r'\b(?:void|double|int|float|char|bool|auto|template<[^>]+>)\s+([A-Za-z_]\w*)\s*\('
    func_def_names = re.findall(func_def_pattern, code)    
    operands = [op for op in operands if op not in func_def_names]
    operands.extend(func_def_names) # Operands solved
    # 2) Adjust Operators
    for func_name in func_def_names:
        # Count how many times this function appears as a definition
        if func_name in operators:
            operators.remove(func_name)  # removes first occurrence only
     
    # String Literals - Only Operands
    double_quotes = re.findall(r'"(?:\\.|[^"\\])*"', code_str, flags=re.DOTALL) 
    single_quotes = re.findall(r"'(?:\\.|[^'\\])+'", code_str, flags=re.DOTALL)  
    operands.extend(double_quotes)
    operands.extend(single_quotes)
    
    # Combine with static subtracting set (C++ keywords, symbols, etc.)
    full_subtracting_set = subtracting_set | dynamic_nonoperands
    operands = [op for op in operands if op not in full_subtracting_set] # Remove non-operands from operands
    plain_logger.debug("Distinct Operators: %s", set(operators))
    plain_logger.debug("Distinct Operands: %s", set(operands))
    n1 = len(set(operators))
    n2 = len(set(operands))
    N1 = len(operators)
    N2 = len(operands)
    return {
        'n1': n1,
        'n2': n2,
        'N1': N1,
        'N2': N2,
    }

def compute_sets(core_non_operands: set, additional_non_operands: set, lang="") -> tuple[str, str, set]:
    """Compute operator and operand regex patterns and the subtracting set for Halstead metrics.

    This ensures deterministic matching by sorting keywords and symbols,
    and escapes special characters to avoid regex issues.

    Args:
        core_non_operands (set): Base set of non-operand keywords (e.g., C++ keywords).
        additional_non_operands (set): Extension-specific non-operand keywords (CUDA, OpenCL, Kokkos, etc.).

    Returns:
        tuple: A tuple containing:
            - operator_pattern (str): Regex pattern for operators.
            - operand_pattern (str): Regex pattern for operands (identifiers and numbers).
            - subtracting_set (set): Set of tokens to exclude from operand count.
    """
    if lang == "cpp":
        operators = core_non_operands
    else:
        operators = core_non_operands | additional_non_operands 

    merged_nonoperands = core_non_operands | additional_non_operands 
    # Split into keyword-like (alphanumeric) and symbolic operators
    keyword_ops = {op for op in operators if re.match(r'^[A-Za-z_]\w*$', op)}
    symbol_ops = operators - keyword_ops
    # Escape keywords and symbols for regex
    escaped_keywords = [r'\b' + re.escape(op) + r'\b' for op in sorted(keyword_ops)] 
    escaped_symbols  = [re.escape(op) for op in sorted(symbol_ops, key=len, reverse=True)]
    escaped_keywords = sorted(escaped_keywords, key=len, reverse=True)
    escaped_symbols  = sorted(escaped_symbols, key=len, reverse=True)
    # Combine patterns: keywords first, then symbols
    operator_pattern = r'|'.join(escaped_keywords + escaped_symbols)
    
    func_call_pattern = r'(?<!::)\b[A-Za-z_]\w*\s*(?=\()' # Don't match namespace called functions Kokkos::
    operator_pattern = f"{operator_pattern}|{func_call_pattern}"
    
    # Operand pattern: identifiers or numeric literals
    operand_pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b' # Doesn't detect string literals
    # Set of non-operands to exclude from operand count
    subtracting_set = merged_nonoperands
    return operator_pattern, operand_pattern, subtracting_set

def halstead_metrics(code: str, lang: str) -> dict:
    "Compute Halstead metrics based on language flag"
    # Mapping from framework name → its non-operands set
    framework_non_operands_map = {
        "cuda": cuda_non_operands,
        "opencl": opencl_non_operands,
        "kokkos": kokkos_non_operands,
        "openmp": openmp_non_operands,
        "adaptivecpp": adaptivecpp_non_operands,
        "openacc": openacc_non_operands,
        "opengl_vulkan": opengl_vulkan_non_operands,
        "webgpu": webgpu_non_operands,
        "boost": boost_non_operands,
        "metal": metal_non_operands,
        "thrust": thrust_non_operands,
        "merged": merged_non_operands,
    }
    detected_langs = []
    if lang == "cpp" or lang == "auto":
        detected_langs = detect_parallel_framework(code)
        # Start with standard C++ non-operands
        auto_non_operands = cpp_non_operands.copy()
        for detected in detected_langs: # can't use for lang in detected_langs -> update lang
            # detected_langs {cpp,cuda} -> non-deterministic ordering such that last iteration
            # could be any one of the 2 values
            if detected != 'cpp' and detected in framework_non_operands_map:
                auto_non_operands |= framework_non_operands_map[detected]
    else:
        auto_non_operands = framework_non_operands_map[lang]
    
    # Standard Code cleaning
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, auto_non_operands, lang)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, lang, detected_langs)

# -------------------------------
# Halstead derived metrics
# -------------------------------
def vocabulary(metrics: dict) -> int:
    """Compute Halstead vocabulary (n = n1 + n2).

    Args:
        metrics (dict): Halstead metrics dictionary.

    Returns:
        int: Vocabulary size.
    """
    return metrics['n1'] + metrics['n2']

def size(metrics: dict) -> int:
    """Compute Halstead program length (N = N1 + N2).

    Args:
        metrics (dict): Halstead metrics dictionary.

    Returns:
        int: Program length.
    """
    return metrics['N1'] + metrics['N2']

def volume(metrics: dict) -> float:
    """Compute Halstead volume (V = N * log2(n)).

    Args:
        metrics (dict): Halstead metrics dictionary.

    Returns:
        float: Program volume.
    """
    return size(metrics) * math.log2(vocabulary(metrics))

def difficulty(metrics: dict) -> float:
    """Compute Halstead difficulty (D = (n1 / 2) * (N2 / n2)).

    Args:
        metrics (dict): Halstead metrics dictionary.

    Returns:
        float: Program difficulty.
    """
    if metrics['n2'] == 0:
        return 0.0
    return (metrics['n1'] / 2) * (metrics['N2'] / metrics['n2'])

def effort(metrics: dict) -> float:
    """Compute Halstead effort (E = D * V).

    Args:
        metrics (dict): Halstead metrics dictionary.

    Returns:
        float: Program effort.
    """
    return difficulty(metrics) * volume(metrics)

def time(metrics: dict) -> float:
    """Compute Halstead estimated time to implement (T = E / 18).

    Args:
        metrics (dict): Halstead metrics dictionary.

    Returns:
        float: Estimated implementation time.
    """
    return effort(metrics) / 18



"""
Edge Cases (General):
1) String Literals are counted as singular operand instances
2) Library calls have to be stripped
3) :: is the scope resolution operator
--> Hence for Kokkos::parallel_for
--> Kokkos and parallel_for are operands — entities being related by ::
**) Function definitions (Operands), but Function invocations (Operators), this would mean the sets n1 and n2
are not strictly disjoint.
**) Constructs: which are neither operators nor operands
4) Use library header for auto language detection -> and only after remove all library headers
"""

"""
Edge Cases (OpenMP):
1) allocateOpenMp<...>()	    - Should be counted as an Operator (function call)
2) GravityEvaluableBase(...)	- Should be counted as an Operator (constructor call)
"""

# Don't want full Function recognitition, since we want CASE-SPECIFIC Parallelizing Framework Construct
# Analysis. Full Function recognition, would analyze ALL functions regardless of parallelizing framework

