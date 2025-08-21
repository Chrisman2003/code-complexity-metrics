import re
import math
from .KEYWORDS import cpp_non_operands, cuda_non_operands, kokkos_non_operands, opencl_non_operands

def halstead_metrics_parametrized(code: str, operator_pattern: str, operand_pattern: str, subtracting_set: set):
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
    # Remove non-operands from operands
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


def compute_sets(core_non_operands: set, additional_non_operands: set) -> tuple[str, str, set]:
    """Compute operator and operand regex patterns and the subtracting set for Halstead metrics.

    This ensures deterministic matching by sorting keywords and symbols,
    and escapes special characters to avoid regex issues.

    Args:
        core_non_operands (set): Base set of non-operand keywords (e.g., C++ keywords).
        additional_non_operands (set): Extension-specific non-operand keywords (CUDA, OpenCL, Kokkos).

    Returns:
        tuple: A tuple containing:
            - operator_pattern (str): Regex pattern for operators.
            - operand_pattern (str): Regex pattern for operands (identifiers and numbers).
            - subtracting_set (set): Set of tokens to exclude from operand count.
    """
    # Merge base and extension-specific non-operators
    merged_nonoperators = core_non_operands | additional_non_operands
    # Split into keyword-like (alphanumeric) and symbolic operators
    keyword_ops = {op for op in merged_nonoperators if re.match(r'^[A-Za-z_]\w*$', op)}
    symbol_ops = merged_nonoperators - keyword_ops
    # Escape keywords and symbols for regex
    escaped_keywords = [r'\b' + re.escape(op) + r'\b' for op in sorted(keyword_ops)]
    escaped_symbols  = [re.escape(op) for op in sorted(symbol_ops, key=len, reverse=True)]
    # Combine patterns: keywords first, then symbols
    operator_pattern = r'|'.join(escaped_keywords + escaped_symbols)
    # Operand pattern: identifiers or numeric literals
    operand_pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b'
    # Set of non-operands to exclude from operand count
    subtracting_set = merged_nonoperators
    return operator_pattern, operand_pattern, subtracting_set


def halstead_metrics_cpp(code: str) -> dict:
    """Compute Halstead metrics for standard C++ code."""
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, set())
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)


def halstead_metrics_cuda(code: str) -> dict:
    """Compute Halstead metrics for CUDA code."""
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, cuda_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)


def halstead_metrics_kokkos(code: str) -> dict:
    """Compute Halstead metrics for Kokkos code."""
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, kokkos_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)


def halstead_metrics_opencl(code: str) -> dict:
    """Compute Halstead metrics for OpenCL code."""
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, opencl_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)


def halstead_metrics_merged(code: str) -> dict:
    """Compute Halstead metrics for a merged set of languages (C++ + GPU extensions)."""
    merged_extensions = cuda_non_operands | opencl_non_operands | kokkos_non_operands
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, merged_extensions)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set)


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
