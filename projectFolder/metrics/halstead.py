import re
import math
from .KEYWORDS import cpp_keywords, cuda_keywords, kokkos_keywords, opencl_keywords

def halstead_metrics_parametrized(code: str, operator_pattern: str, operand_pattern: str, keyword_set: set):
    """
    Parametrized Halstead metrics calculation.
    """
    operators = re.findall(operator_pattern, code)
    operands = re.findall(operand_pattern, code)
    operands = [op for op in operands if op not in keyword_set]
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
    
def halstead_metrics_cpp(code: str):
    operator_pattern = (
        r'\b(?:if|else|switch|case|for|while|do|break|continue|return|goto|try|catch|throw|new|delete|sizeof|typeid|dynamic_cast|static_cast|reinterpret_cast|const_cast|and|or|not|xor|bitand|bitor|compl)\b'
        r'|==|!=|<=|>=|->|->\*|<<=|>>=|<<|>>|&&|\|\||\+\+|--|::|\.|->|[-+*/%=&|^~!]=?|[(){}\[\],;:]'
    )
    operand_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b'
    keyword_set = cpp_keywords
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, keyword_set)

def halstead_metrics_cuda(code: str):
    operator_pattern = (
        r'\b(?:if|else|switch|case|for|while|do|break|continue|return|goto|try|catch|throw|new|delete|sizeof|typeid|dynamic_cast|static_cast|reinterpret_cast|const_cast|and|or|not|xor|bitand|bitor|compl|__global__|__device__|__host__|__shared__|__constant__|__managed__|__restrict__|__threadfence_block|__threadfence|__syncthreads|atomicAdd|atomicSub|atomicExch|atomicMin|atomicMax|atomicInc|atomicDec|atomicCAS|atomicAnd|atomicOr|atomicXor)\b'
        r'|==|!=|<=|>=|->|->\*|<<=|>>=|<<|>>|&&|\|\||\+\+|--|::|\.|->|[-+*/%=&|^~!]=?|[(){}\[\],;:]'
    )
    operand_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b'
    keyword_set = cuda_keywords | cpp_keywords # Merged with C++ keywords
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, keyword_set)

def halstead_metrics_kokkos(code: str):
    operator_pattern = (
        r'\b(?:if|else|switch|case|for|while|do|break|continue|return|goto|try|catch|throw|new|delete|sizeof|typeid|dynamic_cast|static_cast|reinterpret_cast|const_cast|and|or|not|xor|bitand|bitor|compl|Kokkos|KOKKOS_FUNCTION|KOKKOS_INLINE_FUNCTION|KOKKOS_LAMBDA|Kokkos::parallel_for|Kokkos::parallel_reduce|Kokkos::parallel_scan|Kokkos::View|Kokkos::TeamPolicy|Kokkos::RangePolicy|Kokkos::MDRangePolicy)\b'
        r'|==|!=|<=|>=|->|->\*|<<=|>>=|<<|>>|&&|\|\||\+\+|--|::|\.|->|[-+*/%=&|^~!]=?|[(){}\[\],;:]'
    )
    operand_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b'
    keyword_set = kokkos_keywords | cpp_keywords # Merged with C++ keywords
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, keyword_set)

def halstead_metrics_opencl(code: str):
    operator_pattern = (
        r'\b(?:if|else|switch|case|for|while|do|break|continue|return|goto|kernel|__kernel|__global|__local|__constant|__private|get_global_id|get_local_id|get_group_id|get_global_size|get_local_size|get_num_groups|barrier|mem_fence|read_mem_fence|write_mem_fence|CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE)\b'
        r'|==|!=|<=|>=|->|->\*|<<=|>>=|<<|>>|&&|\|\||\+\+|--|::|\.|->|[-+*/%=&|^~!]=?|[(){}\[\],;:]'
    )
    operand_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b'
    keyword_set = opencl_keywords | cpp_keywords # Merged with C++ keywords
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, keyword_set)

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
