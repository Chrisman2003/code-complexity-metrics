# Metrics and statistical pipeline for `code_complexity` package
`code_complexity` is a Python package for analyzing source code complexity metrics, specifically designed for C++ (and extendable with parallel frameworks (GPU)). The `metrics` subpackage provides tools to compute Cognitive, Cyclomatic, Halstead, SLOC, and Nesting Depth complexity. The `stats` subpackage provides tools to process, analyze, and visualize the metrics collected from source code. It complements the `metrics` subpackage by offering a statistical and reporting pipeline for deeper insights into code complexity.

## Package `metrics`
- **Cognitive Complexity**: Regex-based detection of control flow, logical operators, jump statements, and nesting.
- **Cyclomatic Complexity**: Detects functions and computes branch-based complexity.
- **Halstead Metrics**: Computes operators, operands, and related Halstead complexity measures.
- **SLOC (Source Lines of Code)**: Counts logical and physical lines of code.
- **Nesting Depth**: Computes maximum block nesting in source files.

## Package `stats`
- **Analysis**: Computes descriptive statistics and correlations to identify hotspots and relationships between complexity metrics.
- **Report Generator**: Produces PDF reports with tables, plots, and visualizations highlighting complexity of files.
- **Data Loader**: Collects per-file metrics and aggregates GPU-framework complexities for deeper insights.