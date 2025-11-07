import re

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