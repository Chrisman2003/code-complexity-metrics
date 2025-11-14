import re
import math
from .language_tokens import *
from code_complexity.metrics.utils import *
from code_complexity.metrics.cyclomatic import plain_logger

# Substring Suffix Extension Patterns with Respect to the Kleene Operator (*)
# Trailing Commas for Maintainability and minimizing potential errors
# TODO: which languages require :: multiple chaining
pattern_rules = {
    'cuda': [
        r'\b__\w+__\b',                  # CUDA qualifiers like __global__, __device__
        r'\bcuda[A-Z]\w*\b',             # CUDA runtime APIs like cudaMalloc, cudaMemcpy
        r'\batomic[A-Z]\w*\b',           # CUDA atomic intrinsics
    ], # TODO: xif (error != cudaSuccess) {
    'opencl': [
        r'\bcl[A-Z]\w*\b',               # clCreateBuffer, clEnqueueNDRangeKernel
        r'\bget_(?:global|local|group)_id\b',  # non-capturing group
        r'\bcl(?:::\w+)+\b',
    ],
    'kokkos': [ 
        r'\bKokkos::\w+\b',              # All Kokkos namespace calls
    ],
    'openmp': [
        r'\bomp_[a-zA-Z_]+\b',           # matches OpenMP functions like omp_get_num_threads
        r'#pragma\s+omp\s+[a-zA-Z_\s]+', # matches pragmas like #pragma omp parallel for
    ],
    'adaptivecpp': [
        r'\bsycl::\w+\b',                # all SYCL class/function names
        r'\bqueue\b',                    # short forms when using namespace sycl
        r'\bparallel_for\b',
        r'\bsingle_task\b',
        r'\bnd_range\b',
        r'\bnd_item\b',
        # TODO: Matched Operator {'#pragma acc parallel loop\n        for '}
    ],
    'openacc': [
        r'\bacc_\w+\b',                  # OpenACC runtime functions like acc_malloc
        r'#pragma\s+acc\s+[a-zA-Z_\s]+', # OpenACC pragma lines
    ],
    'opengl_vulkan': [
        r'\bvk\w+\b',                    # Vulkan functions like vkCreateInstance
        r'\bgl\w+\b',                    # OpenGL functions like glBindBuffer
        r'\bVK_[A-Z0-9_]+\b',            # Vulkan constants like VK_SUCCESS
        r'\bGL_[A-Z0-9_]+\b',            # OpenGL constants like GL_COMPUTE_SHADER
        r'\bvk::\w+\b',                  # Vulkan C++ API like vk::Instance
        r'\bvk(?:::\w+)+\b',             # Vulkan C++ API like vk::Instance::Instance
        # TODO: Generalized matching for _vk_context ?
    ],
    'webgpu': [
        r'\bwgpu::\w+\b',                # all WebGPU C++ API classes
        r'\bwgpu[A-Z]\w*\b',             # WebGPU runtime functions like wgpuCreateInstance
        r'\bWGPU_[A-Z0-9_]+\b',          # constants
    ],
    'boost': [
        r'\bboost::compute::\w+\b',      # all Boost.Compute classes & functions
        r'\bBOOST_COMPUTE_FUNCTION\b',   # macro
        # TODO: Edge Case:     
        # - namespace compute = boost::compute;
        # - compute::device gpu = compute::system::default_device();
    ],
    'metal': [
        r'\bMTL\w+\b',                   # Metal classes / types
        r'\bMTL::\w+\b',                 # Metal classes / types
        r'\b(device|thread|threadgroup|constant|kernel|sampler|texture)\b',
        r'\bdispatchThreads\b|\bdispatchThreadgroups\b|\bcommit\b|\benqueue\b',
        r'\bnew\w+With\w*:\b',           # Objective-C style method calls
        r'\bMTL_[A-Z0-9_]+\b'            # Metal constants / enums
    ],
    'thrust': [
        r'\bthrust::\w+\b',              # all Thrust API calls and classes
        r'\bTHRUST_[A-Z0-9_]+\b',        # macros
        r'\bthrust(?:::\w+)+\b',         # nested namespaces like thrust::system::cuda
    ], 
    # TODO: non-function constructs detected as functions
    # TODO: edge case - variables named with these prefixations
    # --> HOWEVER, when parall. framework not used, variable naming is RELAXED!
}
"""
    Actual important edge case [after deadline - TODO]:
    -> Arbitrary quantification on occuences of namespace resolution (:: ... :: ... ::)
"""

def halstead_metrics_parametrized(code: str, operator_pattern: str, operand_pattern: str, subtracting_set: set, lang: str, autolist: list[str] = []):
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
    elif lang == "auto":
        for l in autolist:
            if l in pattern_rules:
                patterns_to_apply.extend(pattern_rules[l])
    elif lang in pattern_rules:
        patterns_to_apply = pattern_rules[lang]
    # Result: a list of regex patterns to apply
    # For all regex elements in pattern_rules are used    
    for p_elem in patterns_to_apply:
        matches = re.findall(p_elem, code) # find all matches for this pattern
        operators.extend(matches) # add to operators
        dynamic_nonoperands.update(matches) # add to dynamic non-operands
    
    ## Clean up Operands
    #full_patterns_to_apply = [
    #    pat
    #    for pattern_list in pattern_rules.values()
    #    for pat in pattern_list
    #]
    #for p_elem in full_patterns_to_apply:   # Subtract ALL framework functions from operands 
    #    matches = re.findall(p_elem, code)  # find all matches for this pattern
    #    dynamic_nonoperands.update(matches) # add to dynamic non-operands
            
    # Combine with static subtracting set (C++ keywords, symbols, etc.)
    full_subtracting_set = subtracting_set | dynamic_nonoperands
    double_quotes = re.findall(r'"(?:\\.|[^"\\])*"', code, flags=re.DOTALL) # Multline Kernel String support
    single_quotes = re.findall(r"'(?:\\.|[^'\\])+'", code, flags=re.DOTALL)  
    operands.extend(double_quotes)
    operands.extend(single_quotes)
    
    
    operands = [op for op in operands if op not in full_subtracting_set] # Remove non-operands from operands
    print("Operators", operators)
    print("\n")
    print("Operands", operands)
    print("\n")
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

def detect_parallel_framework(code: str, file_suffix: str = "") -> set[str]:
    """
    Automatically detect the parallelizing framework used in a source file.
    Returns one or a multiple of {'cpp', 'cuda', 'opencl', 'kokkos', 'openmp', 
                    'adaptivecpp', 'openacc', 'opengl_vulkan', 
                    'webgpu', 'boost', 'metal', 'thrust'}
    """
    lib_patterns = { # Assuming correct library declarations
        "cuda": [r'#include\s*<cuda'],
        "opencl": [r'#include\s*<CL/cl[^>]*>'],
        "kokkos": [r'#include\s*<Kokkos'],
        "openmp": [r'#include\s*[<"]omp'],
        "adaptivecpp": [r'#include\s*<CL/sycl'],
        "openacc": [r'#include\s*<openacc'],
        "opengl_vulkan": [r'#include\s*<vulkan'],
        "webgpu": [r'#include\s*<wgpu'],
        "boost": [r'#include\s*"boost'],
        "metal": [r'#include\s*<Metal'],
        "thrust": [r'#include\s*[<"]thrust'],
    }
    if file_suffix == ".slang":
        detected_languages = {"cpp", "slang"}
    else:
        detected_languages = {"cpp"}
    
    for lang, patterns in lib_patterns.items():
        matches = re.findall(patterns[0], code)
        if len(matches) > 0:
            detected_languages.add(lang)
    return detected_languages

def compute_sets(core_non_operands: set, additional_non_operands: set) -> tuple[str, str, set]:
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
    merged_nonoperands = core_non_operands | additional_non_operands 
    #merged_nonoperands = merged_non_operands
    
    # Split into keyword-like (alphanumeric) and symbolic operators
    keyword_ops = {op for op in merged_nonoperands if re.match(r'^[A-Za-z_]\w*$', op)}
    symbol_ops = merged_nonoperands - keyword_ops
    # Escape keywords and symbols for regex
    escaped_keywords = [r'\b' + re.escape(op) + r'\b' for op in sorted(keyword_ops)] 
    escaped_symbols  = [re.escape(op) for op in sorted(symbol_ops, key=len, reverse=True)]
    # Combine patterns: keywords first, then symbols
    operator_pattern = r'|'.join(escaped_keywords + escaped_symbols)
    # Operand pattern: identifiers or numeric literals
    operand_pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b' # Doesn't detect string literals
    # Set of non-operands to exclude from operand count
    subtracting_set = merged_nonoperands
    return operator_pattern, operand_pattern, subtracting_set

def halstead_metrics_cpp(code: str, file_suffix:str = "") -> dict:
    """Compute Halstead metrics for standard C++ code."""
   #
    detected_langs = detect_parallel_framework(code, file_suffix)
    print("Detected Langs: ", detected_langs)
    # Start with standard C++ non-operands
    auto_non_operands = cpp_non_operands
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
        "slang": slang_non_operands, # Header detection impossible
    }
    for lang in detected_langs:
        if lang != 'cpp' and lang in framework_non_operands_map:
            auto_non_operands |= framework_non_operands_map[lang]
            
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)

    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, auto_non_operands)
    # Only for CPP subtract all possible merged non operands
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'auto')


def halstead_metrics_cuda(code: str) -> dict:
    """Compute Halstead metrics for CUDA code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, cuda_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'cuda')


def halstead_metrics_kokkos(code: str) -> dict:
    """Compute Halstead metrics for Kokkos code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, kokkos_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'kokkos')


def halstead_metrics_opencl(code: str) -> dict:
    """Compute Halstead metrics for OpenCL code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, opencl_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'opencl')

def halstead_metrics_openmp(code: str) -> dict:
    """Compute Halstead metrics for OpenMP code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, openmp_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'openmp')

def halstead_metrics_adaptivecpp(code: str) -> dict:
    """Compute Halstead metrics for AdaptiveCPP code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, adaptivecpp_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'adaptivecpp')

def halstead_metrics_openacc(code: str) -> dict:
    """Compute Halstead metrics for OpenACC code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, openacc_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'openacc')

def halstead_metrics_opengl_vulkan(code: str) -> dict:
    """Compute Halstead metrics for OpenGL_Vulkan code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, opengl_vulkan_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'opengl_vulkan')

def halstead_metrics_webgpu(code: str) -> dict:
    """Compute Halstead metrics for WebGPU code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, webgpu_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'webgpu')

def halstead_metrics_boost(code: str) -> dict:
    """Compute Halstead metrics for Boost code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, boost_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'boost')

def halstead_metrics_metal(code: str) -> dict:
    """Compute Halstead metrics for Metal code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, metal_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'metal')

def halstead_metrics_thrust(code: str) -> dict:
    """Compute Halstead metrics for Thrust code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, thrust_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'thrust')

def halstead_metrics_slang(code: str) -> dict:
    """Compute Halstead metrics for Slang code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, slang_non_operands)
    # Slang shader is not embedded in a C++ file
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'slang')


def halstead_metrics_auto(code: str, file_suffix: str = "") -> dict:
    """Compute Halstead metrics with automated language detection for a file (C++ and GPU extensions)."""
    # Example Alleviating edge case
    # Kokkos counts "Kokkos:parallel_for" solely, but with the merged collection
    # The matching pattern "parallel_for" from adaptive cpp would likewise be matched, which is incorrect for exclusive Kokkos
    
    detected_langs = detect_parallel_framework(code, file_suffix)
    # Start with standard C++ non-operands
    auto_non_operands = cpp_non_operands
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
        "slang": slang_non_operands, # Header detection impossible
    }
    # Add non-operands for all detected frameworks (except cpp, which is already included)
    for lang in detected_langs:
        if lang != 'cpp' and lang in framework_non_operands_map:
            auto_non_operands |= framework_non_operands_map[lang]
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, auto_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'auto', detected_langs)

def halstead_metrics_merged(code: str) -> dict:
    """Compute Halstead metrics for a merged set of languages (C++ + GPU extensions)."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    #code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, merged_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'merged')


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


'''
---------------------------------------------------------------
Summary: Why tokenization is not done by whitespace
---------------------------------------------------------------
This analyzer extracts tokens (operators and operands) using
regular expressions rather than splitting on whitespace.   
Reasons:
1. Code tokens in C++ (and its GPU extensions like CUDA, OpenCL,
   Kokkos, etc.) are not reliably separated by spaces.
      Example: "a+b" and "a + b" should yield the same tokens.
2. Operators, punctuation, and symbols (e.g. "==", "->", "::", "{", "}")
   can appear without surrounding whitespace.
3. Comments and string literals must be ignored — they are removed
   before regex extraction.
4. Regex-based matching (via re.findall) provides non-overlapping
   tokens directly, ensuring consistent parsing regardless of spacing. 
Therefore:
Tokenization is pattern-based, not whitespace-based, to correctly
identify syntactic elements across multiple C++-like languages.
---------------------------------------------------------------
'''

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
"""

"""
Edge Cases (OpenMP):
1) allocateOpenMp<...>()	    - Should be counted as an Operator (function call)
2) GravityEvaluableBase(...)	- Should be counted as an Operator (constructor call)
"""

# Don't want full Function recognitition, since we want CASE-SPECIFIC Parallelizing Framework Construct
# Analysis. Full Function recognition, would analyze ALL functions regardless of parallelizing framework

