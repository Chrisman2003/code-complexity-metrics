# Optional: matplotlib/seaborn plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_all_metrics(records, save_dir=None):
    df = pd.DataFrame(records)
    metrics = ["cognitive", "cyclomatic", "nesting", "sloc", "halstead"]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Scatter matrix
    sns.pairplot(df[metrics])
    plt.suptitle("Pairwise Scatter Plots of Complexity Metrics")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pairplot.png"))
        plt.close()
    else:
        plt.show()

    # Correlation heatmap
    corr = df[metrics].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Between Complexity Metrics")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"))
        plt.close()
    else:
        plt.show()

    # Histograms
    for metric in metrics:
        plt.hist(df[metric], bins=20, alpha=0.7)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Count")
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"hist_{metric}.png"))
            plt.close()
        else:
            plt.show()
