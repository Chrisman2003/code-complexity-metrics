"""
Common utility functions used across the repository.

Includes:
- File I/O helpers
- String manipulation
"""
import re
import os
# from code_complexity.metrics.halstead import *

def load_code(filename, TEST_FILES_DIR):
    """Loads the content of a code file.

    Args:
        filename (str): Name of the file to load from TEST_FILES_DIR.

    Returns:
        str: The content of the file as a string.
    """
    with open(os.path.join(TEST_FILES_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()
    

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
    

def remove_cpp_comments(code: str) -> str:
    """
    Remove all C/C++ style comments from the source code.

    This function removes both block comments (`/* ... */`) and 
    single-line comments (`// ...`) from the input code string. 
    The removal is done using regular expressions.

    Args:
        code (str): A string containing C/C++ source code.

    Returns:
        str: The source code with all comments removed.

    Edge Cases:
        - Nested block comments are not supported (C/C++ also do not support nesting). 
        - Strings containing `//` or `/* ... */` inside quotes will not be affected 
          by this function; use `remove_string_literals` first to prevent false positives.
        - Lines containing only comments are removed entirely.
        - Empty input string returns an empty string.
    """
    code_no_block = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL) # Remove all block comments first
    code_no_comments = re.sub(r'//.*', '', code_no_block) # Remove single line comments
    return code_no_comments

def remove_string_literals_old(code: str) -> str:
    """
    Remove all string and character literals from C/C++ source code.

    This function removes:
    - Double-quoted string literals, e.g., "hello world"
    - Single-quoted character literals, e.g., 'a'

    Args:
        code (str): A string containing C/C++ source code.

    Returns:
        str: The source code with all string and character literals removed.

    Edge Cases:
        - Escaped characters inside strings or characters are handled correctly, e.g., 
          "Line\\nBreak" or '\\''.
        - Multi-line string literals are removed.
        - Empty input string returns an empty string.
        - Does not affect comments; use `remove_cpp_comments` to remove comments.
    """
    code = re.sub(r'"(\\.|[^"\\])*"', '', code) # Remove double-quoted strings
    code = re.sub(r"'(\\.|[^'\\])'", '', code) # Remove single-quoted character literals
    return code


'''
__kernel Qualifier: This is an OpenCL C language keyword/qualifier. 
Its purpose is to tell the OpenCL compiler (at runtime) which function 
in the source string is the actual parallel entry point to be executed on the device.
'''
def remove_string_literals(code: str) -> str:
    """Preserve all host-side code and any string literal containing '__kernel', removing all other string literals.

    This function scans the input C/C++/OpenCL source code and selectively keeps string literals
    that contain the '__kernel' keyword, which uniquely identifies OpenCL kernel entry points.
    All other string literals, including normal strings, raw strings, and character literals, 
    are replaced with empty strings.
    Effectively: Any string literal containing the substring __kernel—regardless of 
    what comes before or after—will be kept. All other string literals are stripped.

    Args:
        code (str): Source code containing host and device (kernel) code.

    Returns:
        str: Code with non-kernel string literals replaced by empty strings, kernel strings preserved.
        
    Edge Case:
        - string literal before __kernel is preserved
    """
    def replacer(match: re.Match) -> str:
        s = match.group(0)
        # Keep string literals if they contain '__kernel'
        if "__kernel" in s:
            return s
        # Otherwise, replace with empty string literal
        return '""'
    # Regex pattern to match all string and character literals:
    # 1. R"[\w]*\(.*?\)[\w]*" → matches raw string literals (e.g., R"CLC(...code...)CLC")
    # 2. "(\\.|[^"\\])*" → matches normal double-quoted strings, including escaped characters
    # 3. '(\\.|[^'\\])' → matches single-quoted character literals (e.g., 'a', '\n')
    cleaned_code = re.sub(
        r'R"[\w]*\(.*?\)[\w]*"|"(\\.|[^"\\])*"|\'(\\.|[^\'\\])\'',
        replacer,
        code,
        flags=re.DOTALL
    )
    return cleaned_code

def remove_headers(code: str) -> str:
    """
    Remove all C/C++ style headers inclusions from the source code.

    This functions removes header inclusion calls wrapped in the 
    formats: <...> and "..."
    The removal is performed with regular expressions

    Args:
        code (str): A string containing C/C++ source code.

    Returns:
        str: The source code with all header inclusions removed.

    Edge Cases:
    """
    code = re.sub(r'#\s*include\s*<[^>]*>', '', code)  # removes #include <...>
    code = re.sub(r'#\s*include\s*"[^"]*"', '', code)  # removes #include "..."
    # Edge Cases #include "" and #include <>
    # -> Incorrect Program Behaviour
    # -> First remove headers and then string literals
    return code


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