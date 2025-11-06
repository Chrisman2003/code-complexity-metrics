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
   doc = SimpleDocTemplate(output_path) # PDF document
   styles = getSampleStyleSheet()
   elements = []

   # --- Title ---
   elements.append(Paragraph("<u>Basic Code Complexity Analysis Report</u>", styles['Title']))

   # --- Descriptive Statistics ---
   summary, correlations = summarize(records)
   elements.append(Paragraph("Descriptive Statistics", styles['Heading2']))
   summary_tbl = Table([summary.columns.to_list()] + summary.round(3).values.tolist())
   summary_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightblue),
                                    ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
   elements.append(summary_tbl)
   
   # --- Correlation Matrix ---
   elements.append(Paragraph("Correlation Matrix", styles['Heading2']))
   # Temporary formatting for adding Description column
   row_labels_corr = ["cognitive", "cyclomatic", "nesting", "sloc", "halstead"]
   correlations_table = correlations.copy()
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
   for metric in ['sloc', 'nesting', 'cyclomatic', 'cognitive', 'halstead']:
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
# Helper for GPU delta per file
# -------------------------------
def compute_gpu_delta_for_file(file_path, language):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
        code = f.read()
    return compute_gpu_delta(code, language)

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
   metrics = ['sloc', 'nesting', 'cyclomatic', 'cognitive', 'halstead']    
   # Generate pairplot
   sns.pairplot(df[metrics], diag_kind='hist', corner=True, plot_kws={'color': 'orange'}, diag_kws={'color': 'orange'})
   # Save to a buffer for embedding in PDF
   buf = io.BytesIO()
   plt.savefig(buf, format='png')
   plt.close()
   buf.seek(0) 
   # Add images
   elements.append(Image(buf, width=425, height=425))  # adjust dimensions to fit your layout
    
        
   # --- Build PDF ---
   doc.build(elements)
   print(f"[INFO] Report saved to {output_path}")
    
    
    
   # print(f"🔍 Collecting metrics from {root_dir}...")
   # records = collect_metrics(root_dir)
   # #records = normalize_by_loc(records)   # normalize first
   # df = pd.DataFrame(records)
   # 
   # # Flatten Halstead metrics for easier analysis
   # halstead_keys = ['vocabulary', 'length', 'volume', 'difficulty', 'effort', 'time']
   # halstead_flat = []
   # for rec in records:
   #     hm = rec.get('halstead', {})
   #     flat = {f'halstead_{k}': hm.get(k, 0) for k in halstead_keys}
   #     halstead_flat.append(flat)
   # halstead_df = pd.DataFrame(halstead_flat)
   # df = pd.concat([df, halstead_df], axis=1)
#
   # # Compute GPU delta if requested
   # if compute_gpu_delta:
   #     print("⚡ Computing GPU delta metrics...")
   #     df['gpu_delta'] = df['file_path'].apply(lambda f: compute_gpu_delta_for_file(f, gpu_lang))
   #     # Flatten GPU delta
   #     gpu_delta_flat = pd.DataFrame(df['gpu_delta'].tolist()).add_prefix('gpu_')
   #     df = pd.concat([df, gpu_delta_flat], axis=1)
#
   # # -------------------------------
   # # Distribution Analysis
   # # -------------------------------
   # metrics_cols = ['sloc', 'nesting', 'cyclomatic', 'cognitive'] + [f'halstead_{k}' for k in halstead_keys]
   # if compute_gpu_delta:
   #     metrics_cols += [f'gpu_{k}' for k in halstead_keys]
#
   # print("📊 Plotting distributions...")
   # # Prepare PDF elements
   # doc = SimpleDocTemplate(output_pdf)
   # styles = getSampleStyleSheet()
   # elements = []
   # elements.append(Paragraph("Advanced Code Complexity Report", styles['Title']))
   # elements.append(Spacer(1, 12))
#
   # for metric in metrics_cols:
   #     if metric not in df.columns:
   #         continue
   #     plt.figure(figsize=(6,4))
   #     plt.hist(df[metric], bins=20, color='skyblue', edgecolor='black')
   #     plt.title(f"Distribution of {metric}")
   #     plt.xlabel(metric)
   #     plt.ylabel("Count")
   #     plt.tight_layout()
#
   #     buf = io.BytesIO()
   #     plt.savefig(buf, format='png')
   #     plt.close()
   #     buf.seek(0)
#
   #     elements.append(Image(buf, width=400, height=300))
   #     elements.append(Spacer(1, 12))
#
   # # -------------------------------
   # # Correlation Analysis
   # # -------------------------------
   # corr = df[metrics_cols].corr(method='pearson')
   # elements.append(Paragraph("Correlation Matrix", styles['Heading2']))
   # elements.append(Spacer(1, 6))
   # corr_tbl = Table([corr.columns.to_list()] + corr.round(3).values.tolist())
   # corr_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
   #                               ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
   # 
   # #fit_table_to_page(corr_tbl, doc)
   # elements.append(corr_tbl)
   # elements.append(Spacer(1, 12))
#
   # # -------------------------------
   # # Regression Example: Halstead Volume vs Cyclomatic
   # # -------------------------------
   # if 'halstead_volume' in df.columns and 'cyclomatic' in df.columns:
   #     X = sm.add_constant(df[['cyclomatic']])
   #     y = df['halstead_volume']
   #     model = sm.OLS(y, X).fit()
   #     elements.append(Paragraph("Regression: Halstead Volume ~ Cyclomatic Complexity", styles['Heading2']))
   #     elements.append(Spacer(1, 6))
   #     elements.append(Paragraph(model.summary().as_text().replace("\n", "<br/>"), styles['Code']))
   #     elements.append(Spacer(1, 12))
#
   # # -------------------------------
   # # Clustering / Similarity
   # # -------------------------------
   # print("🔹 Performing KMeans clustering...")
   # scaler = StandardScaler()
   # X_scaled = scaler.fit_transform(df[metrics_cols])
   # kmeans = KMeans(n_clusters=3, random_state=42)
   # df['cluster'] = kmeans.fit_predict(X_scaled)
#
   # elements.append(Paragraph("Clustering of Files (3 clusters)", styles['Heading2']))
   # elements.append(Spacer(1, 6))
   # # Optional: pairplot saved as image
   # plt.figure(figsize=(6,6))
   # sns.pairplot(df, vars=metrics_cols[:4], hue='cluster')  # use first few metrics for visibility
   # buf = io.BytesIO()
   # plt.savefig(buf, format='png')
   # plt.close()
   # buf.seek(0)
   # elements.append(Image(buf, width=400, height=300))
   # elements.append(Spacer(1, 12))
#
   # # -------------------------------
   # # Comparative Analysis: CPU vs GPU
   # # -------------------------------
   # if compute_gpu_delta and 'halstead_volume' in df.columns and 'gpu_volume' in df.columns:
   #     cpu_metrics = df['halstead_volume']
   #     gpu_metrics = df['gpu_volume']
   #     t_stat, p_val = ttest_ind(cpu_metrics, gpu_metrics, equal_var=False)
   #     elements.append(Paragraph(f"GPU vs CPU Halstead Volume t-test: t={t_stat:.2f}, p={p_val:.4f}", styles['Normal']))
   #     elements.append(Spacer(1, 12))
#
   # # -------------------------------
   # # Summary Table
   # # -------------------------------
   # summary = df[metrics_cols].describe().transpose()
   # elements.append(Paragraph("Summary Table", styles['Heading2']))
   # elements.append(Spacer(1, 6))
   # summary_tbl = Table([summary.columns.to_list()] + summary.round(3).values.tolist())
   # summary_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightblue),
   #                                  ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
   # #fit_table_to_page(summary_tbl, doc)
   # elements.append(summary_tbl)
   # elements.append(Spacer(1, 12))
#
   # # -------------------------------
   # # Build PDF
   # # -------------------------------
   # doc.build(elements)
   # print(f"✅ Advanced report saved to {output_pdf}")
