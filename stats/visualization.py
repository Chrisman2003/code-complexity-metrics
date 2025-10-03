# Optional: matplotlib/seaborn plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_all_metrics(records):
    df = pd.DataFrame(records)
    
    # Scatter matrix of numeric metrics
    metrics = ["cognitive", "cyclomatic", "nesting", "sloc"]
    sns.pairplot(df[metrics])
    plt.suptitle("Pairwise Scatter Plots of Complexity Metrics")
    plt.show()
    
    # Correlation heatmap
    corr = df[metrics].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Between Complexity Metrics")
    plt.show()
    
    # Histogram per metric
    for metric in metrics:
        plt.hist(df[metric], bins=20, alpha=0.7)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.show()
