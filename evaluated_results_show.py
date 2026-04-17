import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# LOAD DATA
# =========================
file_path = r"D:\luanvsn\.venv\final\results\model_results.xlsx"
df = pd.read_excel(file_path)

df.columns = df.columns.str.strip()

datasets = df["Dataset"].unique()

output_dir = r"D:\luanvsn\.venv\final\results\dashboard"
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    data = df[df["Dataset"] == dataset]
    models = data["Model"]

    x = np.arange(len(models))

    plt.figure(figsize=(16,10))

    # =========================
    # 1. PERFORMANCE (GIỮ FAF1)
    # =========================
    plt.subplot(2,2,1)
    plt.bar(x - 0.3, data["Accuracy"], width=0.2)
    plt.bar(x - 0.1, data["Precision"], width=0.2)
    plt.bar(x + 0.1, data["Recall"], width=0.2)
    plt.bar(x + 0.3, data["FAF1"], width=0.2)

    plt.xticks(x, models, rotation=45)
    plt.ylim(0.75, 1.0)
    plt.title("Accuracy / Precision / Recall / FAF1")

    # =========================
    # 2. ERROR METRICS
    # =========================
    plt.subplot(2,2,2)
    plt.bar(x - 0.2, data["FPR"], width=0.2)
    plt.bar(x, data["FAR"], width=0.2)
    plt.bar(x + 0.2, data["FAP"], width=0.2)

    plt.xticks(x, models, rotation=45)
    plt.title("FPR / FAR / FAP (Lower is better)")

    # =========================
    # 3. DETECTION RATE
    # =========================
    plt.subplot(2,2,3)
    plt.bar(models, data["DetectionRate"])
    plt.xticks(rotation=45)
    plt.ylim(0.9, 1.0)
    plt.title("Detection Rate")

    # =========================
    # 4. ERROR VIEW
    # =========================
    plt.subplot(2,2,4)
    error = 1 - data["Accuracy"]
    plt.bar(models, error)
    plt.xticks(rotation=45)
    plt.title("Error (1 - Accuracy)")

    # =========================
    plt.suptitle(f"XAI Dashboard - {dataset}", fontsize=16)
    plt.tight_layout()

    # SAVE
    save_path = os.path.join(output_dir, f"{dataset}.png")
    plt.savefig(save_path, dpi=300)

    plt.show()