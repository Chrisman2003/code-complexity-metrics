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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from code_complexity.stats.data_loader import collect_metrics
from code_complexity.gpu_delta import compute_gpu_delta

# -------------------------------
# Helper Functions
# -------------------------------
def compute_gpu_delta_for_file(file_path, language):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
        code = f.read()
    return compute_gpu_delta(code, language)

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
             totals[fw] = {"framework": fw, "halstead_effort": 0.0, "halstead_volume": 0.0, "halstead_difficulty": 0.0}
         totals[fw]["halstead_effort"] += vals.get("effort", 0.0)
         totals[fw]["halstead_volume"] += vals.get("volume", 0.0)
         totals[fw]["halstead_difficulty"] += vals.get("difficulty", 0.0)
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
   df = pd.DataFrame(records) # Convert list of dicts to DataFrame
   for col in ("halstead_volume", "halstead_effort"):
      if col in df.columns:
         df = df.drop(columns=col)
   doc = SimpleDocTemplate(output_path) # PDF document
   styles = getSampleStyleSheet()
   elements = []

   # --- Title ---
   elements.append(Paragraph("<u>Basic Code Complexity Analysis Report</u>", styles['Title']))

   # --- Descriptive Statistics ---
   summary, correlations = summarize(df)
   elements.append(Paragraph("Descriptive Statistics", styles['Heading2']))
   summary_tbl = Table([summary.columns.to_list()] + summary.round(3).values.tolist())
   summary_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightblue),
                                    ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
   elements.append(summary_tbl)
   
   # --- Correlation Matrix ---
   elements.append(Paragraph("Correlation Matrix", styles['Heading2']))
   # Temporary formatting for adding Description column
   correlations_table = correlations.copy()
   row_labels_corr = correlations_table.index.tolist()  # dynamically get labels
   correlations_table['Description'] = row_labels_corr
   cols = ['Description'] + [col for col in correlations_table.columns if col != 'Description']
   correlations_table = correlations_table[cols]
   # Create Table
   corr_tbl = Table([correlations_table.columns.to_list()] + correlations_table.round(3).values.tolist())
   corr_tbl.setStyle(TableStyle([
       ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
       ("GRID", (0,0), (-1,-1), 0.5, colors.black)
   ]))
   elements.append(corr_tbl)
   
   # --- Heat Map ---
   elements.append(Paragraph("Correlation Heatmap", styles['Heading2']))
   plt.figure(figsize=(6, 5))
   sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
   # Save the plot to a BytesIO buffer and embed it
   buf = io.BytesIO() 
   plt.tight_layout() # Adjust layout
   plt.savefig(buf, format='png') # Save to buffer
   plt.close() # Close the plot
   buf.seek(0) # Rewind buffer to the beginning
   # Add images
   elements.append(Image(buf, width=345, height=260)) # display size of rendered image in pdf

   # --- Boxplots + Histograms Side by Side ---
   elements.append(Paragraph("Boxplots + Histograms", styles['Heading2']))
   for metric in ['sloc', 'nesting', 'cyclomatic', 'cognitive', 'halstead_difficulty']:
      # Create a figure with 2 subplots side by side
      _, axes = plt.subplots(ncols=2, figsize=(10, 4))  # 2 columns
      # Histogram
      axes[0].hist(df[metric], bins=20, color='lightblue', edgecolor='black')
      axes[0].set_title(f"Histogram of {metric}")
      axes[0].set_xlabel(metric)
      axes[0].set_ylabel("Count")
      # Boxplot
      sns.boxplot(y=df[metric], color='#4682B4', ax=axes[1])
      axes[1].set_title(f"Boxplot of {metric}")
      axes[1].set_ylabel(metric)
      plt.tight_layout()
      # Save the combined figure to buffer
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      plt.close()
      buf.seek(0)
      # Add image to PDF
      elements.append(Image(buf, width=560, height=210))  # adjust width/height to fit layout
       
   # --- Build PDF ---
   doc.build(elements)
   print(f"[INFO] Report saved to {output_path}")
    
# -------------------------------
# Advanced Statistical Analysis: 3 Page Report
# -------------------------------
'''
- Segregated Analysis on GPU-Delta
- Halstead Functions: 
--> Volume
--> Difficulty
--> Effort
'''
def generate_advanced_report(records, output_path="complexity_report.pdf"):
   """
   Generate a PDF report with:
     1. Descriptive statistics
     2. Pairwise Plot
     3. T tests
   """
   df = pd.DataFrame(records) # Convert list of dicts to DataFrame
   doc = SimpleDocTemplate(output_path) # PDF document
   styles = getSampleStyleSheet()
   elements = []

   # --- Title ---
   elements.append(Paragraph("<u>Advanced Code Complexity Analysis Report</u>", styles['Title']))

   # --- Descriptive Statistics ---
   summary, correlations = summarize(records)
   elements.append(Paragraph("Descriptive Statistics", styles['Heading2']))
   summary_tbl = Table([summary.columns.to_list()] + summary.round(3).values.tolist())
   summary_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.orange), # Orange Color: #FD5000
                                    ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
   elements.append(summary_tbl)
   
   # --- Pairplot ---
   elements.append(Paragraph("Pairwise Scatter Plot (Pairplot)", styles['Heading2']))
   metrics = ['sloc', 'nesting', 'cyclomatic', 'cognitive', 'halstead_difficulty']    
   # Generate pairplot
   sns.pairplot(df[metrics], diag_kind='hist', corner=True, plot_kws={'color': 'orange'}, diag_kws={'color': 'orange'})
   # Save to a buffer for embedding in PDF
   buf = io.BytesIO()
   plt.savefig(buf, format='png')
   plt.close()
   buf.seek(0) 
   # Add images
   elements.append(Image(buf, width=425, height=425))  # adjust dimensions to fit your layout
    
   # --- Aggregated Framework Plot ---
   elements.append(Paragraph("Framework-Wise Aggregated Complexity (Difficulty vs Volume, size = Effort)", styles['Heading2']))

   # Aggregate per framework
   df_fw = pd.DataFrame(aggregate_framework_complexity(records))

   if not df_fw.empty:
       plt.figure(figsize=(7, 6))
       palette = sns.color_palette("husl", len(df_fw))
       fw_color_map = {fw: palette[i] for i, fw in enumerate(df_fw["framework"])}

       # Normalize bubble sizes (effort → circle area)
       eff_scaled = (df_fw["halstead_effort"] / df_fw["halstead_effort"].max()) * 2000

       plt.scatter(
           df_fw["halstead_difficulty"],
           df_fw["halstead_volume"],
           s=eff_scaled,  # bubble size = effort
           c=[fw_color_map[fw] for fw in df_fw["framework"]],
           edgecolors="black",
           alpha=0.8
       )

       for i, row in df_fw.iterrows():
           plt.text(row["halstead_difficulty"], row["halstead_volume"], row["framework"], fontsize=9, ha="center", va="center")

       plt.title("Aggregated GPU Framework Complexity", fontsize=12)
       plt.xlabel("Halstead Difficulty")
       plt.ylabel("Halstead Volume")
       plt.tight_layout()

       buf = io.BytesIO()
       plt.savefig(buf, format='png')
       plt.close()
       buf.seek(0)
       elements.append(Image(buf, width=580, height=530))
   else:
       elements.append(Paragraph("No GPU frameworks detected for aggregation.", styles['Normal']))

   # --- Build PDF ---
   doc.build(elements)
   print(f"[INFO] Report saved to {output_path}")
    
    
    