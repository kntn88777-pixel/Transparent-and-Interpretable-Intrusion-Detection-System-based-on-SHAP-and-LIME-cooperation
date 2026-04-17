import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================================
# 1. LOAD DATA
# =========================================================
file_path = r"D:\luanvsn\.venv\final\results\evaluated_results.xlsx"
df = pd.read_excel(file_path)

# =========================================================
# 2. OUTPUT FOLDER
# =========================================================
output_dir = "visualization_results"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 3. DEFINE METRICS
# =========================================================
performance_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "DetectionRate",
    "FAF1"
]

error_metrics = [
    "FPR",
    "FAR",
    "FAP"
]

time_metrics = [
    "TrainTime"
]

all_metrics = performance_metrics + error_metrics + time_metrics

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# =========================================================
# 4. VẼ 9 BIỂU ĐỒ RIÊNG LẺ
# =========================================================
print("Generating individual plots...")

for metric in all_metrics:

    plt.figure(figsize=(12,6))

    if metric == "TrainTime":
        sns.lineplot(
            data=df,
            x="Model",
            y=metric,
            hue="Dataset",
            marker="o",
            linewidth=2.5
        )
    else:
        sns.barplot(
            data=df,
            x="Model",
            y=metric,
            hue="Dataset"
        )

    plt.title(f"{metric} Comparison Across Models and Datasets", fontsize=14)
    plt.xticks(rotation=30)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# =========================================================
# 5. FIGURE LỚN 1: PERFORMANCE METRICS
# =========================================================
print("Generating grouped performance figure...")

fig, axes = plt.subplots(2, 3, figsize=(20,10))
axes = axes.flatten()

for i, metric in enumerate(performance_metrics):
    sns.barplot(
        data=df,
        x="Model",
        y=metric,
        hue="Dataset",
        ax=axes[i]
    )
    axes[i].set_title(metric)
    axes[i].tick_params(axis='x', rotation=30)

# Remove empty subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Figure_Performance_Metrics.png"), dpi=300)
plt.close()

# =========================================================
# 6. FIGURE LỚN 2: ERROR METRICS
# =========================================================
print("Generating grouped error figure...")

fig, axes = plt.subplots(1, 3, figsize=(18,5))

for i, metric in enumerate(error_metrics):
    sns.barplot(
        data=df,
        x="Model",
        y=metric,
        hue="Dataset",
        ax=axes[i]
    )
    axes[i].set_title(metric)
    axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Figure_Error_Metrics.png"), dpi=300)
plt.close()

# =========================================================
# 7. FIGURE LỚN 3: TRAIN TIME
# =========================================================
print("Generating TrainTime figure...")

plt.figure(figsize=(12,6))

sns.lineplot(
    data=df,
    x="Model",
    y="TrainTime",
    hue="Dataset",
    marker="o",
    linewidth=3
)

plt.title("Training Time Comparison", fontsize=15)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "Figure_TrainTime.png"), dpi=300)
plt.close()

print("="*50)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print(f"Saved in folder: {output_dir}")
print("="*50)