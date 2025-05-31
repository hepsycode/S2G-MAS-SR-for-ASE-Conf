import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("Metrics.csv")

# Metrics to plot
metrics = ["Levenshtein", "Cosine", "Precision", "Recall", "F1"]

# Model groups by prefix
model_groups = {
    "G2-": "Gemma2-2b",
    "LL31-": "Llama3.1",
    "LL32-": "Llama3.2",
    "ML-": "Mistral Large",
    "M8-": "Mistral 8x7B"
}

# Color palette per group
group_labels = list(model_groups.values())
palette_colors = sns.color_palette("Set2", n_colors=len(group_labels))
group_palette = dict(zip(group_labels, palette_colors))

# Assign model group to each row
def assign_group(target):
    for prefix, group in model_groups.items():
        if target.startswith(prefix):
            return group
    return "Other"

df["ModelGroup"] = df["Target"].apply(assign_group)

# Set seaborn style
sns.set(style="whitegrid")

# Plot loop
for metric in metrics:
    plt.figure(figsize=(6, 6)) # 6,6
    ax = sns.violinplot(
        data=df,
        x="Target",
        y=metric,
        hue="ModelGroup",
        palette=group_palette,
        cut=0,
        dodge=False,
        legend=False
    )

    # Y axis starts at 0
    ax.set_ylim(bottom=0)

    # Gridlines
    ax.yaxis.grid(True, which='major', linestyle='--', color='black', linewidth=0.8)
    ax.yaxis.grid(True, which='minor', linestyle=':', color='gray', linewidth=0.5)
    ax.minorticks_on()

    # Axis titles in bold
    ax.set_xlabel("Target", fontsize=24, fontweight='bold')
    ax.set_ylabel(metric, fontsize=24, fontweight='bold')

    # Chart title in bold
    # ax.set_title(f"Distribution of {metric} by Target", fontsize=14, fontweight='bold')

    # Set spines (borders) to black
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    # Tick label styling (bold and size 14)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Add vertical lines at X positions
    xticks = ax.get_xticks()
    ylim = ax.get_ylim()

    xticks = ax.get_xticks()
    ylim = ax.get_ylim()

    for x in xticks:
        ax.axvline(x=x, ymin=0, ymax=1, linestyle='--', color='lightgray', linewidth=0.8)

        # Optional: add a small "tick" below the x-axis
        ax.plot([x], [ylim[0]], marker='|', markersize=10, color='gray', clip_on=False)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"img/violinplot_{metric}.png", bbox_inches='tight')
    plt.close()
