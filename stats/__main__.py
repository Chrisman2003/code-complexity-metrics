from stats.data_loader import collect_metrics
from stats.preprocessing import normalize_by_loc
from stats.analysis import summarize
from stats.visualization import plot_all_metrics

def main():
    # 1️⃣ Collect metrics from your samples
    root_dir = "samples/"  # adjust path as needed
    print("Collecting metrics...")
    records = collect_metrics(root_dir)
    print(f"Collected metrics for {len(records)} files.\n")
    
    # 2️⃣ Preprocess / normalize metrics
    records = normalize_by_loc(records)
    print("Metrics normalized by LOC.\n")
    
    # 3️⃣ Statistical analysis
    summary, correlations = summarize(records)
    print("Descriptive statistics:\n", summary, "\n")
    print("Correlations:\n", correlations, "\n")
    
    # 4️⃣ Visualization
    plot_all_metrics(records)
    
if __name__ == "__main__":
    main()
