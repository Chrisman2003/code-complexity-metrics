import pandas as pd
import matplotlib.pyplot as plt
from stats.analysis import summarize
from stats.visualization import plot_all_metrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os
"""
report_generator.py
Generate statistical + visual reports for code complexity analysis.
"""

def generate_report(records, output_path="complexity_report.pdf"):
    """
    Generate a PDF report with:
      1. Descriptive statistics
      2. Correlation analysis
      3. Statistical tests (ANOVA, regressions, etc.)
      4. Plots and visualizations
    """
    df = pd.DataFrame(records)
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    elements = []

    # --- Title ---
    elements.append(Paragraph("Code Complexity Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # --- Descriptive Statistics ---
    summary, correlations = summarize(records)
    elements.append(Paragraph("Descriptive Statistics", styles['Heading2']))
    elements.append(Spacer(1, 6))
    summary_tbl = Table([summary.columns.to_list()] + summary.round(3).values.tolist())
    summary_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightblue),
                                     ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
    elements.append(summary_tbl)
    elements.append(Spacer(1, 12))

    # --- Correlations ---
    elements.append(Paragraph("Correlation Matrix", styles['Heading2']))
    elements.append(Spacer(1, 6))
    corr_tbl = Table([correlations.columns.to_list()] + correlations.round(3).values.tolist())
    corr_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                                  ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
    elements.append(corr_tbl)
    elements.append(Spacer(1, 12))

    # --- Statistical Tests ---
    #elements.append(Paragraph("Statistical Tests", styles['Heading2']))
    #stats_results = run_statistical_tests(df)
    #for name, result in stats_results.items():
    #    elements.append(Paragraph(f"<b>{name}:</b> {result}", styles['Normal']))
    #    elements.append(Spacer(1, 6))

    # --- Plots (saved and embedded) ---
    elements.append(Paragraph("Visualizations", styles['Heading2']))

    # Example: save plots and embed them
    os.makedirs("report_figs", exist_ok=True)
    plot_all_metrics(records, save_dir="report_figs")
    for plot_file in os.listdir("report_figs"):
        elements.append(Image(os.path.join("report_figs", plot_file), width=400, height=300))
        elements.append(Spacer(1, 12))

    # --- Build PDF ---
    doc.build(elements)
    print(f"[INFO] Report saved to {output_path}")
