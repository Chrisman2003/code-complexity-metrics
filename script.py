#import matplotlib.pyplot as plt

# Data
#algorithms = ['vector_addition', 'nbody_simulation', 'matrix_multiplication', 'polyhedral']
#SLOC = [53, 163, 39, 252]
#Volume = [3607.57, 12229.22, 3645.84, 23156.30]
#Difficulty = [82.58, 205.12, 97.85, 336.35]

## Scale SLOC for circle size (adjust factor for readability)
#sizes = [s * 3 for s in SLOC]  

## Scatter plot
#plt.figure(figsize=(10,6), dpi=1000)
#plt.scatter(Volume, Difficulty, s=sizes, alpha=0.6, c='teal', edgecolors='black')

## Annotate points
#for i, alg in enumerate(algorithms):
    #plt.text(Volume[i]*1.01, Difficulty[i]*1.01, alg, fontsize=10)

#plt.xlabel('Halstead Volume')
#plt.ylabel('Halstead Difficulty')
#plt.title('Halstead Volume vs Difficulty (circle size ~ SLOC)')
#plt.grid(True, linestyle='--', alpha=0.5)
## Save figure as PNG
#plt.savefig('halstead_volume_vs_difficulty.png', dpi=500)
# import matplotlib.pyplot as plt
# 
# # Algorithm names
# algorithms = ["Vector Addition", "N-Body Simulation", "Matrix Multiplication", "Polyhedral Gravity"]
# 
# # Cyclomatic Complexity (x-axis)
# cyclomatic = [3, 18, 3, 55]
# 
# # Cognitive Complexity (y-axis)
# cognitive = [0, 9, 1, 68]
# 
# # Nesting Depth (used for circle size)
# nesting_depth = [4, 5, 4, 8]
# 
# # Scale nesting depth for circle size (adjust factor for readability)
# sizes = [nd * 100 for nd in nesting_depth]  # 100 is scaling factor, tweak as needed
# 
# # Scatter plot
# plt.figure(figsize=(8,6))
# plt.scatter(cyclomatic, cognitive, s=sizes, color='darkblue', alpha=0.6, edgecolors='black')
# 
# # Annotate points with algorithm names
# for i, alg in enumerate(algorithms):
#     plt.text(cyclomatic[i]+0.5, cognitive[i]+0.5, alg, fontsize=10)
# 
# plt.xlabel("Cyclomatic Complexity")
# plt.ylabel("Cognitive Complexity")
# plt.title("Cyclomatic vs Cognitive Complexity of Kokkos Implementations\n(circle size ~ Nesting Depth)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('Cognitive_Cyclomatic.png', dpi=500)

import matplotlib.pyplot as plt

# Example data for each algorithm
# Fill in with actual metrics from your output
# Each entry: ("Implementation", Volume, Difficulty, SLOC)

matrix_multiplication = [
    ("Cpp", 2636.63, 88.44, 37),
    ("OpenCL", 6417.08, 122.00, 84),
    ("Boost", 3519.05, 75.66, 41),
    ("Kokkos", 3645.84, 97.85, 39),
    ("Cuda", 10347.18, 215.36, 118),
    ("OpenMP", 4122.19, 113.50, 50),
    ("Vulkan", 3029.28, 63.66, 31),
]

polyhedral_gravity_model = [
    ("Kokkos", 23156.30, 336.35, 252),
    ("Cuda", 36866.92, 517.33, 419),
    ("AdaptiveCpp", 28828.13, 381.19, 308),
    ("OpenCL", 13652.73, 176.76, 175),
    ("OpenMP", 24115.32, 402.17, 275),
    ("OpenACC", 23487.35, 366.92, 271),
    ("WebGPU", 18464.71, 250.93, 266),
    ("OpenGL_Vulkan", 38149.62, 643.53, 440),
    ("CPP_128", 23760.89, 361.21, 286),
]

nbody_simulation = [
    ("Cpp", 6146.00, 119.89, 80),
    ("OpenCL", 14529.58, 224.20, 166),
    ("Boost", 10958.49, 156.62, 121),
    ("Kokkos", 12229.22, 205.12, 163),
    ("Cuda", 16479.41, 275.83, 199),
    ("OpenMP", 12846.52, 233.33, 159),
    ("Vulkan", 9551.16, 187.94, 93),
    ("AdaptiveCpp", 13828.81, 217.83, 162),
]

vector_addition = [
    ("Cpp", 2180.11, 56.61, 38),
    ("OpenACC", 2619.75, 68.76, 41),
    ("OpenCL", 6679.99, 115.16, 95),
    ("Boost", 4068.02, 107.02, 57),
    ("Kokkos", 3607.57, 82.58, 53),
    ("Metal", 5735.68, 100.71, 80),
    ("Cuda", 2907.60, 135.07, 54),
    ("OpenMP", 2395.94, 62.13, 37),
    ("Vulkan", 3887.81, 78.18, 53),
    ("AdaptiveCpp", 3898.24, 81.61, 55),
]

#
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#
#def safe_filename(name):
#    return name.replace(" ", "_").replace("-", "_")
#
#def plot_metrics(data, algorithm_name):
#    names, volumes, difficulties, slocs = zip(*data)
#    sizes = [s * 15 for s in slocs]  # scale factor for visibility
#
#    # Assign a unique color to each framework
#    frameworks = list(set(names))
#    colors_list = list(mcolors.TABLEAU_COLORS.values())  # reliable distinct colors
#    if len(frameworks) > len(colors_list):
#        colors_list += list(mcolors.CSS4_COLORS.values())  # fallback if more frameworks
#
#    color_map = {fw: colors_list[i % len(colors_list)] for i, fw in enumerate(frameworks)}
#    point_colors = [color_map[name] for name in names]
#
#    plt.figure(figsize=(10, 6))
#    scatter = plt.scatter(volumes, difficulties, s=sizes, c=point_colors, alpha=0.8, edgecolor='k')
#
#    # Legend for frameworks
#    for fw in frameworks:
#        plt.scatter([], [], color=color_map[fw], label=fw)
#    plt.legend(title="Framework", bbox_to_anchor=(1.05, 1), loc='upper left')
#
#    plt.xlabel("Halstead Volume")
#    plt.ylabel("Halstead Difficulty")
#    plt.title(f"{algorithm_name}: Difficulty vs Volume (Framework color, SLOC size)")
#    plt.grid(True)
#    plt.tight_layout()
#    plt.savefig(safe_filename(algorithm_name) + ".png", dpi=600)
#
## Generate plots
#plot_metrics(matrix_multiplication, "Matrix Multiplication")
#plot_metrics(nbody_simulation, "N-Body Simulation")
#plot_metrics(polyhedral_gravity_model, "Polyhedral Gravity Model")
#plot_metrics(vector_addition, "Vector Addition")
#

import pandas as pd

# Combine all data into a single DataFrame
all_data = []

for alg_name, data in [
    ("Matrix Multiplication", matrix_multiplication),
    ("Polyhedral Gravity Model", polyhedral_gravity_model),
    ("N-Body Simulation", nbody_simulation),
    ("Vector Addition", vector_addition),
]:
    for fw, vol, diff, sloc in data:
        all_data.append({
            "Algorithm": alg_name,
            "Framework": fw,
            "SLOC": sloc,
            "Halstead Volume": vol,
            "Halstead Difficulty": diff,
        })

df = pd.DataFrame(all_data)

# Optionally sort for readability
df.sort_values(by=["Algorithm", "Halstead Volume"], ascending=[True, False], inplace=True)

# Display as a table in Jupyter / Python
print(df)

# Export as LaTeX for your thesis
latex_table = df.to_latex(index=False, float_format="%.2f")
with open("framework_complexity_table.tex", "w") as f:
    f.write(latex_table)

# If you want to visualize a big table on top of the plots
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
tbl = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.auto_set_column_width(col=list(range(len(df.columns))))
plt.tight_layout()
plt.savefig("Frameworks_Algorithmic", dpi = 600)
