import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("Consistency.csv")

# Metrics to plot
metrics = ["String Consistency", "Papers Consistency"]

# Model groups by prefix
model_groups = {
    "Gemma2-2b": "Gemma2-2b",
    "Llama3.1-8B": "Llama3.1",
    "Mistral-Large": "Mistral Large",
    "mixtral-8x7B": "Mistral 8x7B"
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

df["ModelGroup"] = df["LLM"].apply(assign_group)

# Set seaborn style
sns.set(style="whitegrid")

# Plot loop
for metric in metrics:
    plt.figure(figsize=(6, 6))

    # Compute Q1, median, Q3
    grouped = df.groupby("LLM")[metric]
    q1 = grouped.quantile(0.25)
    median = grouped.median() # quantile(0.50) # grouped.quantile(0.50)
    q3 = grouped.quantile(0.75)

    # Compute medians and IQR
    median_vals = df.groupby("LLM")[metric].median()
    iqr_vals = df.groupby("LLM")[metric].quantile(0.75) - df.groupby("LLM")[metric].quantile(0.25)

    # Compute asymmetric error bars
    lower_err = (iqr_vals / 4) #  (median - q1) / 2
    upper_err = (iqr_vals / 4) #  (q3 - median) / 2
    # lower_err = (median - q1) / 2
    # upper_err = (q3 - median) / 2

    # Create plot DataFrame
    plot_df = pd.DataFrame({
        "LLM": median.index,
        "Median": median.values,
        "LowerError": lower_err.values,
        "UpperError": upper_err.values
    })
    plot_df["ModelGroup"] = plot_df["LLM"].apply(assign_group)

    # Cap errors
    plot_df["UpperError"] = plot_df.apply(
        lambda row: min(row["UpperError"], 1 - row["Median"]),
        axis=1
    )
    plot_df["LowerError"] = plot_df["LowerError"].clip(lower=0)

    # Draw barplot with no border first
    # colors = plot_df["ModelGroup"].map(group_palette)
    ax = sns.barplot(
        data=plot_df,
        x="LLM",
        y="Median",
        color='lightgray',
        #hue="ModelGroup",
        #palette=group_palette,
        #dodge=False  # Important! So that bars stay aligned per LLM
    )

    # Apply facecolor and edge manually for each bar
    for i, bar in enumerate(ax.patches):
        group = plot_df.iloc[i]["ModelGroup"]
        color = group_palette.get(group, "gray")
        bar.set_facecolor(color)
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)

    # Add custom asymmetric error bars
    for i, row in plot_df.iterrows():
        ax.errorbar(
            x=i,
            y=row["Median"],
            yerr=[[row["LowerError"]], [row["UpperError"]]],
            fmt='none',
            ecolor='black',
            capsize=5,
            linewidth=1
        )

    # Style
    #ax.set_ylim(bottom=0)
    # ax.set_ylim(bottom=0.001, top=1)  # Important! Cannot set 0 for log scale, set a small positive number like 0.001
    # ax.set_yscale('log')  # Set Y axis to log scale

    # Define custom Y ticks
    # custom_ticks = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    # ax.set_yticks(custom_ticks)
    # ax.set_yticklabels([str(tick) for tick in custom_ticks])  # show as normal numbers

    ax.yaxis.grid(True, which='major', linestyle='--', color='black', linewidth=0.8)
    ax.yaxis.grid(True, which='minor', linestyle=':', color='gray', linewidth=0.5)
    ax.minorticks_on()
    ax.set_xlabel("Target", fontsize=22, fontweight='bold')
    ax.set_ylabel(metric, fontsize=22, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Add vertical lines at X positions
    xticks = ax.get_xticks()
    ylim = ax.get_ylim()

    for x in xticks:
        ax.axvline(x=x, ymin=0, ymax=1, linestyle='--', color='lightgray', linewidth=0.8)

        # Optional: add a small "tick" below the x-axis
        ax.plot([x], [ylim[0]], marker='|', markersize=10, color='gray', clip_on=False)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"img/barplot_{metric}.png", bbox_inches='tight')
    plt.close()