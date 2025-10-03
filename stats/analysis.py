# Core statistical computations using pandas/scipy
import pandas as pd

def summarize(records):
    df = pd.DataFrame(records)

    # Descriptive stats
    summary = df.describe()
    
    # Correlations between metrics
    correlations = df[["cognitive", "cyclomatic", "nesting", "sloc"]].corr()
    
    # Optional: Halstead effort correlation
    if "halstead_effort_per_line" in df.columns:
        correlations["halstead_effort"] = df["halstead_effort_per_line"].corr(df["cognitive_per_line"])
    
    return summary, correlations
