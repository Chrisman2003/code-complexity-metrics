# -----------------------------------------------------------------------------
# Common Utility Functions for C++ and GPU-Extended Code Analysis
# -----------------------------------------------------------------------------
# Includes:
# - File I/O helpers for loading source code from specified directories
# - Parallel framework detection (C++, CUDA, OpenCL, Kokkos, OpenMP, SYCL, OpenACC, Vulkan/OpenGL, WebGPU, Boost, Metal, Thrust)
# - Source code preprocessing utilities:
#     * Removal of C++ headers (#include <...>, #include "...")
#     * Removal of C++-style comments (block /* */ and single-line //)
#     * Selective removal of string literals while preserving OpenCL __kernel strings
#
# Note:
# These functions provide a foundation for metrics computation (Halstead, cyclomatic, SLOC, etc.)
# by ensuring that non-executable text and library-specific constructs do not interfere.
# Preprocessing order is important: headers and comments should be removed before analyzing
# string literals to prevent false positives.
# -----------------------------------------------------------------------------
import re
import os
import logging
import sys

"""
LOGGING Metrics
"""
metrics_logger = logging.getLogger("metrics")
metrics_handler = logging.StreamHandler(sys.stdout) 
metrics_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
))
metrics_logger.addHandler(metrics_handler)
metrics_logger.setLevel(logging.INFO)

"""
LOGGING Values
"""
plain_logger = logging.getLogger("plain")
plain_handler = logging.StreamHandler(sys.stdout)
plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain_logger.addHandler(plain_handler)
plain_logger.setLevel(logging.INFO)


def load_code(filename, TEST_FILES_DIR):
    """Loads the content of a code file.

    Args:
        filename (str): Name of the file to load from TEST_FILES_DIR.

    Returns:
        str: The content of the file as a string.
    """
    with open(os.path.join(TEST_FILES_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()
    

def detect_parallel_framework(code: str) -> set[str]:
    """Detects the parallelizing frameworks used in a source code file.

    This function scans the source code for library includes corresponding to common 
    parallel programming frameworks and returns a set of detected languages.
    "cpp" is always included as a base language.

    Args:
        code (str): The source code as a string.

    Returns:
        set[str]: A set of strings representing the detected frameworks. Possible
                  values include:
                  {"cpp", "cuda", "opencl", "kokkos", "openmp", "adaptivecpp",
                   "openacc", "opengl_vulkan", "webgpu", "boost", "metal", "thrust"}.
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
    detected_languages = {"cpp"}
    
    for lang, patterns in lib_patterns.items():
        matches = re.findall(patterns[0], code)
        if len(matches) > 0:
            detected_languages.add(lang)
    plain_logger.debug("Detected Languages: %s", detected_languages)
    return detected_languages    
    

def remove_cpp_comments(code: str) -> str:
    """
    Remove all C++ style comments from the source code.

    This function removes both block comments (`/* ... */`) and 
    single-line comments (`// ...`) from the input code string. 
    The removal is done using regular expressions.

    Args:
        code (str): A string containing C++ source code.

    Returns:
        str: The source code with all comments removed.

    Edge Cases:
        - Nested block comments are not supported (C++ also does not support nesting). 
        - Strings containing `//` or `/* ... */` inside quotes will not be affected 
          by this function; use `remove_string_literals` first to prevent false positives.
        - Lines containing only comments are removed entirely.
        - Empty input string returns an empty string.
    """
    code_no_block = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL) # Remove all block comments first
    code_no_comments = re.sub(r'//.*', '', code_no_block) # Remove single line comments
    return code_no_comments

def remove_string_literals(code: str) -> str:
    """Preserve all host-side code and any string literal containing '__kernel', removing all other string literals.

    This function scans the input C++ source code and selectively keeps string literals
    that contain the '__kernel' keyword, which uniquely identifies OpenCL kernel entry points.
    All other string literals, including normal strings, raw strings, and character literals, 
    are replaced with empty strings.
    Effectively: Any string literal containing the substring '__kernel' regardless of 
    what comes before or after will be kept. All other string literals are stripped.

    Args:
        code (str): Source code containing host and device (kernel) code.

    Returns:
        str: Code with non-kernel string literals replaced by empty strings, kernel strings preserved.
        
    Edge Case:
        - string literal before '__kernel' is preserved
    """
    def replacer(match: re.Match) -> str:
        s = match.group(0)
        # Keep string literals if they contain '__kernel'
        if "__kernel" in s:
            return s
        # Otherwise, replace with empty string literal
        return '""'
    # Regex pattern to match all string and character literals:
    cleaned_code = re.sub(
        r'R"[\w]*\(.*?\)[\w]*"|"(\\.|[^"\\])*"|\'(\\.|[^\'\\])\'',
        replacer,
        code,
        flags=re.DOTALL
    )
    return cleaned_code

def remove_headers(code: str) -> str:
    """
    Remove all C++ style headers from the source code.

    This functions removes header inclusion calls wrapped in the 
    formats: <...> and "..."
    The removal is performed with regular expressions

    Args:
        code (str): A string containing C++ source code.

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