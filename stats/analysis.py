# Core statistical computations using pandas/scipy
import pandas as pd

def summarize(records):
    df = pd.DataFrame(records)

    # Descriptive stats
    summary = df.describe()
    
    # Correlations between metrics
    correlations = df[["cognitive", "cyclomatic", "nesting", "sloc", "halstead"]].corr()
       
    return summary, correlations
