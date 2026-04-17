import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================
# 1. LOAD FILE
# =====================================================
file_path = r"D:\luanvsn\.venv\final\metric1\FINAL_AVG_BY_DATASET.xlsx"
df = pd.read_excel(file_path)

# =====================================================
# 2. OUTPUT FOLDER
# =====================================================
output_dir = "xai_metric_plots"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# 3. XAI METHODS
# =====================================================
xai_methods = [
    "KernelSHAP",
    "TreeSHAP",
    "DeepSHAP",
    "LIME"
]

# =====================================================
# 4. METRIC LIST
# =====================================================
metrics = df["Metric"].unique()

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# =====================================================
# 5. VẼ 10 BIỂU ĐỒ RIÊNG
# =====================================================
for metric in metrics:

    metric_df = df[df["Metric"] == metric]

    plot_df = metric_df.melt(
        id_vars=["Dataset", "Metric"],
        value_vars=xai_methods,
        var_name="XAI_Method",
        value_name="Score"
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=plot_df,
        x="Dataset",
        y="Score",
        hue="XAI_Method"
    )

    plt.title(f"{metric} Comparison Across XAI Methods", fontsize=14)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{metric}.png"),
        dpi=300
    )
    plt.close()

# =====================================================
# 6. FIGURE TỔNG HỢP 10 SUBPLOTS
# =====================================================
fig, axes = plt.subplots(4, 3, figsize=(20,18))
axes = axes.flatten()

for i, metric in enumerate(metrics):

    metric_df = df[df["Metric"] == metric]

    plot_df = metric_df.melt(
        id_vars=["Dataset", "Metric"],
        value_vars=xai_methods,
        var_name="XAI_Method",
        value_name="Score"
    )

    sns.barplot(
        data=plot_df,
        x="Dataset",
        y="Score",
        hue="XAI_Method",
        ax=axes[i]
    )

    axes[i].set_title(metric)
    axes[i].tick_params(axis='x', rotation=20)

# Remove unused subplots if any
for j in range(len(metrics), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "All_XAI_Metrics.png"),
    dpi=300
)
plt.close()

print("="*50)
print("DONE! All XAI metric plots saved to:", output_dir)
print("="*50)