from code_complexity.metrics.halstead import *
'''How much additional “syntactic complexity” is introduced just by GPU constructs.'''

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
    else:
        raise ValueError(f"Unsupported language: {language}")

    # Compute per-metric delta
    delta = {key: gpu_metrics[key] - cpp_metrics.get(key, 0) for key in gpu_metrics}
    return delta
