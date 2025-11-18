import pandas as pd

def summarize(records):
    # Descriptive stats
    df = pd.DataFrame(records)
    summary = df.describe()
    
    # Base columns for correlation
    base_cols = ["cognitive", "cyclomatic", "nesting", "sloc", "halstead_effort"]
    
    # Add optional halstead columns if they exist
    if "halstead_volume" in df.columns:
        base_cols.append("halstead_volume")
    if "halstead_difficulty" in df.columns:
        base_cols.append("halstead_difficulty")
        
    # Compute correlations only for columns that exist in df
    correlations = df[[col for col in base_cols if col in df.columns]].corr()
    row_labels_summ = ["File count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    summary['Description'] = row_labels_summ
    cols = ['Description'] + [col for col in summary.columns if col != 'Description']
    summary = summary[cols]

    return summary, correlations
'''
Example Records Format:
records = [
    {"sloc": 100, "cyclomatic": 10, "cognitive": 5, "nesting": 2, "halstead": 200},
    {"sloc": 150, "cyclomatic": 12, "cognitive": 6, "nesting": 3, "halstead": 250},
    {"sloc": 90, "cyclomatic": 8, "cognitive": 4, "nesting": 1, "halstead": 180}
]
'''