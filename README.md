# code-complexity-metrics
Bachelor's thesis project for implementing and comparing code complexity metrics to assess developer productivity - Christian Ispas

Getting Started
Follow these steps to set up and run the code-complexity-metrics project on your machine:

1. Clone the Repository:
Clone the project from GitHub using the following command:
Bash
git clone "https://github.com/Chrisman2003/code-complexity-metrics.git"
When prompted, enter your GitHub username and your Personal Access Token (as password authentication is deprecated).

2. Change into the Project Directory:
Navigate into the newly cloned project directory:
Bash
cd code-complexity-metrics/

3. Create and Activate a Virtual Environment:
It's highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts with system-wide Python packages.
Bash
python3 -m venv .venv
source .venv/bin/activate
You'll know the virtual environment is active when (.venv) appears at the beginning of your terminal prompt.

4. Install Project Dependencies:
With the virtual environment active, install the project's dependencies, including the project itself in editable mode:
Bash
pip install -e .
This will install all required packages listed in the pyproject.toml file, including networkx, pyopencl, numpy, and others.

5. Pull the Latest Changes (Optional but Recommended):
To ensure you have the most up-to-date code, pull the latest changes from the feature/OpenCL-support branch:
Bash
git pull
You might be prompted for your GitHub username and Personal Access Token again.

6. Run the code-metrics Tool:
Now you can execute the code-metrics tool. For example, to analyze tests/test_files/complex.cpp:
Bash
code-metrics tests/test_files/complex.cpp