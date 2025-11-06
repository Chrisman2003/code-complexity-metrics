import pandas as pd

def summarize(records):
    df = pd.DataFrame(records) # Convert list of dicts to DataFrame
    # Descriptive stats
    summary = df.describe()
    # Correlations between metrics
    correlations = df[["cognitive", "cyclomatic", "nesting", "sloc", "halstead"]].corr()
    # corr() in pandas computes pairwise correlations between numeric columns.
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