import pandas as pd
import matplotlib.pyplot as plt
from code_complexity.stats.analysis import summarize
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import seaborn as sns
import io
from scipy.stats import ttest_ind, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
from code_complexity.stats.data_loader import collect_metrics
from code_complexity.gpu_delta import compute_gpu_delta
from matplotlib.lines import Line2D

# -------------------------------
# Helper Functions
# -------------------------------
def compute_gpu_delta_for_file(file_path, language):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
        code = f.read()
    return compute_gpu_delta(code, language)

def plot_to_image(elements, width, height):
    """
    Saves the current matplotlib figure into a BytesIO buffer
    and appends it as a ReportLab Image to the PDF elements list.
    """
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    elements.append(Image(buf, width=width, height=height))

def aggregate_framework_complexity(records):
    """
    Aggregate GPU complexity metrics across all files,
    grouped by framework (excluding cpp).
    Returns a list of dicts with one row per framework.
    """    
    totals = {}
    for rec in records:
        gpu_data = rec.get("gpu_complexity", {})
        for fw, vals in gpu_data.items():
            if fw == "cpp":
                continue
            if fw not in totals:
                totals[fw] = {
                    "Framework": fw,
                    "Halstead_Effort": 0.0,
                    "Halstead_Volume": 0.0,
                    "Halstead_Difficulty": 0.0
                }
            totals[fw]["Halstead_Effort"] += vals.get("effort", 0.0)
            totals[fw]["Halstead_Volume"] += vals.get("volume", 0.0)
            totals[fw]["Halstead_Difficulty"] += vals.get("difficulty", 0.0)
    return list(totals.values())

# -------------------------------
# Basic Statistical Analysis: 3 Page Report
# -------------------------------
def generate_basic_report(records, output_path="complexity_report.pdf"):
    """
    Generate a PDF report with:
    1. Descriptive statistics
    2. Correlation analysis
    3. Histograms and Boxplots
    """
    df = pd.DataFrame(records)
    # Remove if present
    for col in ("halstead_volume", "halstead_effort"):
        if col in df.columns:
            df = df.drop(columns=col)
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    elements = []

    # --- Title ---
    elements.append(Paragraph("<u>Basic Code Complexity Analysis Report</u>", styles['Title']))

    # --- Descriptive Statistics ---
    summary, correlations = summarize(df)
    elements.append(Paragraph("Descriptive Statistics", styles['Heading2']))
    summary_tbl = Table([summary.columns.to_list()] + summary.round(3).values.tolist())
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(summary_tbl)

    # --- Correlation Matrix ---
    elements.append(Paragraph("Correlation Matrix", styles['Heading2']))
    correlations_table = correlations.copy()
    row_labels_corr = correlations_table.index.tolist()
    correlations_table['Description'] = row_labels_corr
    cols = ['Description'] + [c for c in correlations_table.columns if c != 'Description']
    correlations_table = correlations_table[cols]
    corr_tbl = Table([correlations_table.columns.to_list()] +
                     correlations_table.round(3).values.tolist())
    corr_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(corr_tbl)

    # --- Heat Map ---
    elements.append(Paragraph("Correlation Heatmap", styles['Heading2']))
    plt.figure(figsize=(6, 5), dpi=400)
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.xticks(rotation=90)  # 90 degrees → vertical
    plt.yticks(rotation=0)   # 0 degrees → horizontal
    plot_to_image(elements, width=350, height=285) # Buf IO Handle

    # --- Boxplots + Histograms ---
    elements.append(Paragraph("Boxplots + Histograms", styles['Heading2']))
    for metric in ['sloc', 'nesting', 'cyclomatic', 'cognitive', 'halstead_difficulty']:
        _, axes = plt.subplots(ncols=2, figsize=(10, 4), dpi=400)
        axes[0].hist(df[metric], bins=20, color='lightblue', edgecolor='black')
        axes[0].set_title(f"Histogram of {metric}")
        axes[0].set_xlabel(metric)
        axes[0].set_ylabel("Count")
        
        sns.boxplot(y=df[metric], color='#4682B4', ax=axes[1])
        axes[1].set_title(f"Boxplot of {metric}")
        axes[1].set_ylabel(metric)
        plt.tight_layout()
        plot_to_image(elements, width=560, height=210) # Buf IO Handle
    
    # --- Build PDF ---
    doc.build(elements)
    print(f"[INFO] Report saved to {output_path}")


# -------------------------------
# Advanced Statistical Analysis: 3 Page Report
# -------------------------------
def generate_advanced_report(records, output_path="complexity_report.pdf"):
    """
    Generate a PDF report with:
    1. Descriptive statistics
    2. Pairwise plot
    3. Framework analysis
    4. T tests
    """
    df = pd.DataFrame(records)
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    elements = []

    # --- Title ---
    elements.append(Paragraph("<u>Advanced Code Complexity Analysis Report</u>", styles['Title']))

    # --- Descriptive Statistics ---
    summary, correlations = summarize(records)
    elements.append(Paragraph("Descriptive Statistics", styles['Heading2']))
    summary_tbl = Table([summary.columns.to_list()] + summary.round(3).values.tolist())
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.orange),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(summary_tbl)

    # --- Pairplot ---
    elements.append(Paragraph("Pairwise Scatter Plot (Pairplot)", styles['Heading2']))
    metrics = ['sloc', 'nesting', 'cyclomatic', 'cognitive', 'halstead_difficulty']
    sns.pairplot(df[metrics], diag_kind='hist', corner=True,
                 plot_kws={'color': 'orange'}, diag_kws={'color': 'orange'})
    plot_to_image(elements, width=425, height=425) # Buf IO Handle

    # --- Framework Aggregated Plot ---
    elements.append(Paragraph(  
        "GPU-Framework Native Halstead Complexity",
        styles['Heading2']
    ))
    df_fw = pd.DataFrame(aggregate_framework_complexity(records))
    # Table
    gpu_tbl = Table([df_fw.columns.to_list()] + df_fw.round(2).values.tolist())
    gpu_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.orange),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(gpu_tbl)
    # Plot
    plt.figure(figsize=(7, 6), dpi=400)
    palette = sns.color_palette("husl", len(df_fw)) 
    fw_color_map = {fw: palette[i] for i, fw in enumerate(df_fw["Framework"])} # Distinct colors
    # Plot Bubbles
    for fw in df_fw["Framework"]: # fw is a framework name
        fw_row = df_fw[df_fw["Framework"] == fw] # Select Row based on a predicate
        plt.scatter(
            fw_row["Halstead_Difficulty"],
            fw_row["Halstead_Volume"],
            s=fw_row["Halstead_Effort"] / df_fw["Halstead_Effort"].max() * 2000, # Normalize Bubble Size
            c=[fw_color_map[fw]],
            edgecolors="black",
            alpha=0.8,
        )
    # Custom legend handles with uniform marker size
    frameworks = list(df_fw["Framework"])
    handles = [
        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor=fw_color_map[fw],
               markersize=10,
               markeredgecolor='black',
               alpha=0.8)
        for fw in frameworks
    ]
    plt.legend(handles=handles, labels=frameworks, title="Framework",
               bbox_to_anchor=(1, 0), loc='lower right')
    plt.title("Halstead Effort = Circle Diameter", fontsize=12)
    plt.xlabel("Halstead Difficulty")
    plt.ylabel("Halstead Volume")
    plt.tight_layout()
    plot_to_image(elements, width=580, height=450) # Buf IO Handle
    
    # --- Build PDF ---
    doc.build(elements)
    print(f"[INFO] Report saved to {output_path}")


# TODO: Pure Addition of Halstead Metrics may not be meaningful