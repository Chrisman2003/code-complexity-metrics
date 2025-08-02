import re
import math

def basic_halstead_metrics(code: str):
    '''
    n1: number of unique operators 
    n2: number of unique operands
    N1: Total number of operators
    N2: Total number of operands
    '''
    operator_pattern = r'\b(?:if|else|elif|for|while|try|except|return|and|or|not)\b|==|!=|<=|>=|[-+*/%=<>&|^~!]=?|[(){}\[\],.:]'   
    # () = function call
    # [] = list or indexing
    # , = argument separation
    # : = start of code block
    # . = attribute/method access
    operand_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b'
    operators = re.findall(operator_pattern, code)
    operands = re.findall(operand_pattern, code)
    
    keywords = set([
        'if', 'else', 'elif', 'for', 'while', 'return', 'try', 'except', 'finally', 'with', # Evaluation operators    
        'def', 'class', 'break', 'continue', 'pass', 'import', 'from', 'as', 'lambda', 'yield', # Function and class definitions
        'in', 'is', 'not', 'and', 'or', 'print', 'True', 'False', 'None', # Boolean and None
        '-', '+', '*', '/', '%', '=', '==', '!=', '<=', '>=', '<', '>', '&', '|', '^', '~', '!', # Bitwise operators
        '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=', '>>', '<<', # Assignment operators
        '(', ')', '{', '}', '[', ']', ',', '.', ':', ';', '#', '@', '->', '=>', '::', '...', # Punctuation and special characters
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