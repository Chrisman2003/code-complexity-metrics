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
            dynamic_nonoperands.update(matches) # add to dynamic non-operands
        else:
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
    #print("Operators", operators)
    #print("\n")
    #print("Operands", operands)
    #print("\n")
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
    merged_nonoperands = core_non_operands | additional_non_operands 
    
    if lang == "cpp":
        operators = core_non_operands
    else:
        operators = merged_nonoperands

    # Split into keyword-like (alphanumeric) and symbolic operators
    keyword_ops = {op for op in operators if re.match(r'^[A-Za-z_]\w*$', op)}
    symbol_ops = operators - keyword_ops
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
    detected_langs = detect_parallel_framework(code, file_suffix)
    # Start with standard C++ non-operands
    
    #print("local cpp non operand state", cpp_non_operands)
    auto_non_operands = cpp_non_operands.copy()
    # auto_non_operands = cpp_non_operands
    
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
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, auto_non_operands, "cpp")
    # Only for CPP subtract all possible merged non operands
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, 'cpp', detected_langs)


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
    auto_non_operands = cpp_non_operands.copy()
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

import re
from code_complexity.metrics.utils import *
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
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code = remove_string_literals(code)
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
        complexity += stripped.count('?') * (1 + nesting)  # Ternary operator: can't nest structures inside it
        for op in logical_ops_alpha:
            matches = re.findall(rf'\b{op}\b', stripped) # Word boundaries to avoid false positives
            complexity += len(matches) * (1 + nesting)
            
        nesting = max(0, nesting - stripped.count('}'))
        # Increase nesting ONLY if a nesting keyword and a brace are on this line.
        #found_nesting_keyword = False
        if (stripped != '{' and stripped != ""): # NEW
            found_nesting_keyword = False # NEW
        for keyword in control_keywords + ['->']:
            if re.search(rf'\b{keyword}\b', stripped): # Word boundaries to avoid false positives
                found_nesting_keyword = True
                break
        if found_nesting_keyword and '{' in stripped: # No nesting for if (n == 0) return 1.0;
            nesting += stripped.count('{')    
            found_nesting_keyword = False # NEW
            
    return complexity

'''
for ()
    for ()
'''
 

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
    [x] 8) Null-Coalescing Operators -> Doesn't exist in C++
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


# --- OLD ---
'''
def enrich_metrics(base_metrics: dict) -> dict:
    """Add derived Halstead metrics (vocabulary, size, volume, difficulty, effort, time)."""
    enriched = dict(base_metrics)  # shallow copy
    enriched["vocabulary"] = vocabulary(base_metrics)
    enriched["size"] = size(base_metrics)
    enriched["volume"] = volume(base_metrics)
    enriched["difficulty"] = difficulty(base_metrics)
    enriched["effort"] = effort(base_metrics)
    enriched["time"] = time(base_metrics)
    return enriched

def compute_gpu_delta(code: str, language: str) -> dict:
    """
    Compute GPU delta metrics for a given code snippet in the specified language.

    :param code: The source code to analyze.
    :param language: The programming language of the code ('cuda', 'opencl', 'kokkos').
    :return: A dictionary containing the GPU delta metrics (GPU - C++).
    """
    cpp_metrics = enrich_metrics(halstead_metrics_cpp(code))

    if language == "cuda":
        gpu_metrics = enrich_metrics(halstead_metrics_cuda(code))
    elif language == "opencl":
        gpu_metrics = enrich_metrics(halstead_metrics_opencl(code))
    elif language == "kokkos":
        gpu_metrics = enrich_metrics(halstead_metrics_kokkos(code))
    elif language == "adaptivecpp":
        gpu_metrics = enrich_metrics(halstead_metrics_adaptivecpp(code))
    elif language == "openacc":
        gpu_metrics = enrich_metrics(halstead_metrics_openacc(code))
    elif language == "opengl_vulkan":
        gpu_metrics = enrich_metrics(halstead_metrics_opengl_vulkan(code))
    elif language == "webgpu":
        gpu_metrics = enrich_metrics(halstead_metrics_webgpu(code))
    elif language == "boost":
        gpu_metrics = enrich_metrics(halstead_metrics_boost(code))
    elif language == "metal":
        gpu_metrics = enrich_metrics(halstead_metrics_metal(code))
    elif language == "thrust":
        gpu_metrics = enrich_metrics(halstead_metrics_thrust(code))
    elif language == "auto":
        gpu_metrics = enrich_metrics(halstead_metrics_auto(code))
    elif language == "merged":
        gpu_metrics = enrich_metrics(halstead_metrics_merged(code))
    else:
        raise ValueError(f"Unsupported language: {language}")

'''

def halstead_metrics_cpp(code: str, file_suffix:str = "") -> dict:
    """Compute Halstead metrics for standard C++ code."""
    detected_langs = detect_parallel_framework(code, file_suffix)
    # Start with standard C++ non-operands
    auto_non_operands = cpp_non_operands.copy()
    
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
    }
    for lang in detected_langs:
        if lang != 'cpp' and lang in framework_non_operands_map:
            auto_non_operands |= framework_non_operands_map[lang]
            
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, auto_non_operands, "cpp")
    # Only for CPP subtract all possible merged non operands
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, "cpp", detected_langs)


def halstead_metrics_cuda(code: str) -> dict:
    """Compute Halstead metrics for CUDA code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, cuda_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'cuda')


def halstead_metrics_kokkos(code: str) -> dict:
    """Compute Halstead metrics for Kokkos code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, kokkos_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'kokkos')


def halstead_metrics_opencl(code: str) -> dict:
    """Compute Halstead metrics for OpenCL code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, opencl_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'opencl')

def halstead_metrics_openmp(code: str) -> dict:
    """Compute Halstead metrics for OpenMP code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, openmp_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'openmp')

def halstead_metrics_adaptivecpp(code: str) -> dict:
    """Compute Halstead metrics for AdaptiveCPP code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, adaptivecpp_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'adaptivecpp')

def halstead_metrics_openacc(code: str) -> dict:
    """Compute Halstead metrics for OpenACC code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, openacc_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'openacc')

def halstead_metrics_opengl_vulkan(code: str) -> dict:
    """Compute Halstead metrics for OpenGL_Vulkan code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, opengl_vulkan_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'opengl_vulkan')

def halstead_metrics_webgpu(code: str) -> dict:
    """Compute Halstead metrics for WebGPU code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, webgpu_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'webgpu')

def halstead_metrics_boost(code: str) -> dict:
    """Compute Halstead metrics for Boost code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, boost_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'boost')

def halstead_metrics_metal(code: str) -> dict:
    """Compute Halstead metrics for Metal code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, metal_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'metal')

def halstead_metrics_thrust(code: str) -> dict:
    """Compute Halstead metrics for Thrust code."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, thrust_non_operands)
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'thrust')

def halstead_metrics_auto(code: str, file_suffix: str = "") -> dict:
    """Compute Halstead metrics with automated language detection for a file (C++ and GPU extensions)."""
    # Example Alleviating edge case
    # Kokkos counts "Kokkos:parallel_for" solely, but with the merged collection
    # The matching pattern "parallel_for" from adaptive cpp would likewise be matched, which is incorrect for exclusive Kokkos
    
    detected_langs = detect_parallel_framework(code, file_suffix)
    # Start with standard C++ non-operands
    auto_non_operands = cpp_non_operands.copy()
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
    }
    # Add non-operands for all detected frameworks (except cpp, which is already included)
    for lang in detected_langs:
        if lang != 'cpp' and lang in framework_non_operands_map:
            auto_non_operands |= framework_non_operands_map[lang]
    #print(detected_langs)
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, auto_non_operands)
    
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'auto', detected_langs)

def halstead_metrics_merged(code: str) -> dict:
    """Compute Halstead metrics for a merged set of languages (C++ + GPU extensions)."""
    code = remove_headers(code)
    code = remove_cpp_comments(code)
    code_str = code # Strings are immutable
    code = remove_string_literals(code)
    operator_pattern, operand_pattern, subtracting_set = compute_sets(cpp_non_operands, merged_non_operands)
    
    return halstead_metrics_parametrized(code, operator_pattern, operand_pattern, subtracting_set, code_str, 'merged')

import os
from code_complexity.metrics.sloc import *
from code_complexity.metrics.nesting_depth import *
from code_complexity.metrics.cyclomatic import *
from code_complexity.metrics.cognitive import *
from code_complexity.metrics.halstead import *
from code_complexity.metrics.utils import detect_parallel_framework
from pathlib import Path

'''
✅ Ranked from most to least accurate for total file complexity:
1) E — Effort
2) D — Difficulty
3) V — Volume
'''
def collect_metrics(root_path: str):
    """Scan a file or directory and compute complexity metrics."""
    records = []

    def analyze_file(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        sloc_val = compute_sloc(code)
        nesting = compute_nesting_depth(code)
        cyclomatic_val = regex_compute_cyclomatic(code)
        cognitive_val = regex_compute_cognitive(code)
        halstead_difficulty = difficulty(halstead_metrics_auto(code, Path(filepath).suffix))
        halstead_volume = volume(halstead_metrics_auto(code, Path(filepath).suffix))
        halstead_effort = effort(halstead_metrics_auto(code, Path(filepath).suffix))
        
        # Base Halstead metrics (C++ reference)
        halstead_base = halstead_metrics_cpp(code, Path(filepath).suffix)
        halstead_difficulty_base = difficulty(halstead_base)
        halstead_volume_base = volume(halstead_base)
        
        # Detect languages/frameworks
        languages = detect_parallel_framework(code, Path(filepath).suffix)
        # Mapping of language/framework to metric function
        lang_to_fn = {
            'cpp': halstead_metrics_cpp,
            'cuda': halstead_metrics_cuda,
            'kokkos': halstead_metrics_kokkos,
            'opencl': halstead_metrics_opencl,
            'openmp': halstead_metrics_openmp,
            'adaptivecpp': halstead_metrics_adaptivecpp,
            'openacc': halstead_metrics_openacc,
            'opengl_vulkan': halstead_metrics_opengl_vulkan,
            'webgpu': halstead_metrics_webgpu,
            'boost': halstead_metrics_boost,
            'metal': halstead_metrics_metal,
            'thrust': halstead_metrics_thrust,
        }

        # Compute GPU-native Halstead complexities
        gpu_complexity = {}
        for lang in languages:
            if lang in lang_to_fn and lang != 'cpp':  # skip base C++
                halstead_lang = lang_to_fn[lang](code)
                gpu_complexity[lang] = {
                    "difficulty": difficulty(halstead_lang) - halstead_difficulty_base,
                    "volume": volume(halstead_lang) - halstead_volume_base,
                }
        
        return {
            "file": filepath,
            "sloc": sloc_val,
            "nesting": nesting,
            "cognitive": cognitive_val,
            "cyclomatic": cyclomatic_val,
            # Halstead Metric
            "halstead_difficulty": halstead_difficulty,
            "halstead_volume": halstead_volume,
            "halstead_effort": halstead_effort,
            "gpu_complexity": gpu_complexity,
        }
    # Handle single file
    if os.path.isfile(root_path) and root_path.endswith((".cpp", ".cu", ".slang")):
        records.append(analyze_file(root_path))
    # Handle directory
    elif os.path.isdir(root_path):
        for subdir, _, files in os.walk(root_path):
            for file in files:
                if file.endswith((".cpp", ".cu", ".slang")):
                    filepath = os.path.join(subdir, file)
                    records.append(analyze_file(filepath))
    return records

'''
    "file": "example.cu",
    "sloc": 120,
    "nesting": 3,
    "cognitive": 15,
    "cyclomatic": 8,
    "halstead_difficulty": 20,
    "halstead_volume": 1500,
    "gpu_complexity": {
        "cuda": {"difficulty": 10, "volume": 600},
        "openmp": {"difficulty": 5, "volume": 100}
    }
'''