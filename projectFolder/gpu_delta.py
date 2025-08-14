from projectFolder.metrics.halstead import *

def compute_gpu_delta(code:str, language:str) -> dict:
    """
    Compute GPU delta metrics for a given code snippet in the specified language.
    
    :param code: The source code to analyze.
    :param language: The programming language of the code ('cpp', 'cuda', 'opencl', 'kokkos').
    :return: A dictionary containing the GPU delta metrics.
    """
    if language == 'cpp':
        return halstead_metrics_cpp(code) - halstead_metrics_cpp(code)
    elif language == 'cuda':
        return halstead_metrics_cuda(code) - halstead_metrics_cpp(code)
    elif language == 'opencl':
        return halstead_metrics_opencl(code) - halstead_metrics_cpp(code)
    elif language == 'kokkos':
        return halstead_metrics_kokkos(code) - halstead_metrics_cpp(code)
    else:
        raise ValueError(f"Unsupported language: {language}")