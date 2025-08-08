import re
import math

def basic_halstead_metrics(code: str):
    '''
    n1: number of unique operators 
    n2: number of unique operands
    N1: Total number of operators
    N2: Total number of operands
    '''
    # C++ keywords and operators
    operator_pattern = (
        r'\b(?:if|else|switch|case|for|while|do|break|continue|return|goto|try|catch|throw|new|delete|sizeof|typeid|dynamic_cast|static_cast|reinterpret_cast|const_cast|and|or|not|xor|bitand|bitor|compl)\b'
        r'|==|!=|<=|>=|->|->\*|<<=|>>=|<<|>>|&&|\|\||\+\+|--|::|\.|->|[-+*/%=&|^~!]=?|[(){}\[\],;:]'
    )
    # () = function call, [] = array, {} = block, , = argument separation, ; = statement end, : = initializer list, :: = scope resolution, . = member access, -> = pointer member access
    operand_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b'
    operators = re.findall(operator_pattern, code)
    operands = re.findall(operand_pattern, code)
    
    keywords = set([
        # C++ keywords
        'if', 'else', 'switch', 'case', 'for', 'while', 'do', 'break', 'continue', 'return', 'goto', 'try', 'catch', 'throw',
        'new', 'delete', 'sizeof', 'typeid', 'dynamic_cast', 'static_cast', 'reinterpret_cast', 'const_cast',
        'and', 'or', 'not', 'xor', 'bitand', 'bitor', 'compl', 'true', 'false', 'nullptr',
        'int', 'float', 'double', 'char', 'void', 'short', 'long', 'signed', 'unsigned', 'bool', 'class', 'struct', 'union', 'enum', 'namespace', 'public', 'private', 'protected', 'virtual', 'template', 'typename', 'using', 'static', 'const', 'volatile', 'mutable', 'explicit', 'inline', 'friend', 'operator', 'this', 'extern', 'register', 'auto', 'thread_local', 'static_assert', 'constexpr', 'decltype', 'export', 'import', 'module', 'requires', 'concept', 'co_await', 'co_return', 'co_yield', 'asm', 'default', 'override', 'final', 'noexcept', 'nullptr_t', 'type', 'wchar_t', 'char16_t', 'char32_t',
        # C++ operators and punctuation
        '-', '+', '*', '/', '%', '=', '==', '!=', '<=', '>=', '<', '>', '&', '|', '^', '~', '!', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=', '>>', '<<', '&&', '||', '++', '--', '->', '->*', '.', '::', '(', ')', '{', '}', '[', ']', ',', '.', ':', ';', '#', '@', '...', '?'
    ])    
    
    operands = [op for op in operands if op not in keywords]
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
    
'''
Formulas for Halstead Metrics:
- n = Vocabulary = n1 + n2
- N = Size = N1 + N2
- V = Volume = N*log2(n)
- D = Difficulty = (n1/2) * (N2/n2)
- E = Effort = D * V
- T = Time = E / 18
# 18 is the arbitrary default value for the stroud number
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