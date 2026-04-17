import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# ────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
PLOT_DIR = Path("plots_eval")
os.makedirs(PLOT_DIR / "confusion", exist_ok=True)
os.makedirs(PLOT_DIR / "roc", exist_ok=True)
RESULTS_FILE = "results/evaluated_results.xlsx"

# Dataset cần đánh giá
DATASET_FILES = [
    {"name": "CIC17", "X_test": "preprocessed_cic17_X_test.npy", "Y_test": "preprocessed_cic17_y_test.npy"},
    {"name": "NSL-KDD", "X_test": "preprocessed_nsl-kdd_X_test.npy", "Y_test": "preprocessed_nsl-kdd_y_test.npy"},
    {"name": "SIGNAL2022", "X_test": "signal22_X_test.npy", "Y_test": "signal22_y_test.npy"},
]

# Danh sách tất cả mô hình muốn đánh giá (đã thêm đầy đủ)
MODELS_TO_EVAL = ["AdaBoost", "RandomForest", "LightGBM", "MLP", "DNN", "CNN", "LSTM"]


# ────────────────────────────────────────────────
# METRICS
# ────────────────────────────────────────────────
def compute_ids_metrics(y_true, y_pred):
    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        FAR = FP / (FP + TN) if (FP + TN) != 0 else 0
        FAP = FP / (FP + TP) if (FP + TP) != 0 else 0
        FAF1 = (2 * TP) / (2 * TP + FN + 2 * FP) if (2 * TP + FN + 2 * FP) != 0 else 0
        DR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    else:
        FAR = FAP = FAF1 = DR = FPR = 0
    return FAR, FAP, FAF1, DR, FPR


def plot_confusion_matrix(cm, dataset, model_name):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{dataset} – {model_name} Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion" / f"{dataset}_{model_name}_cm.png", dpi=120)
    plt.close()


def plot_roc(y_true, y_score, dataset, model_name):
    y_true_bin = (y_true != 0).astype(int)
    if y_score.ndim == 2 and y_score.shape[1] > 1:
        y_score_bin = y_score[:, 1:].max(axis=1)
    else:
        y_score_bin = y_score.ravel()
    y_score_bin = np.clip(y_score_bin, 0, 1)

    fpr, tpr, _ = roc_curve(y_true_bin, y_score_bin)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{dataset} – {model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc" / f"{dataset}_{model_name}_roc.png", dpi=150)
    plt.close()


# ────────────────────────────────────────────────
# LOAD TEST DATA
# ────────────────────────────────────────────────
def load_test_data(ds):
    print(f"\nLoading test data for: {ds['name']}")
    X_test = np.load(DATA_DIR / ds["X_test"])
    y_test = np.load(DATA_DIR / ds["Y_test"])
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    num_classes = len(np.unique(y_test))
    print(f"   X_test shape: {X_test.shape} | Classes: {num_classes}")
    return X_test, y_test, num_classes


# ────────────────────────────────────────────────
# MAIN EVALUATION
# ────────────────────────────────────────────────
def main():
    results = []

    for ds in DATASET_FILES:
        print("\n" + "=" * 70)
        print(f"Evaluating dataset: {ds['name']}")
        X_test, y_test, num_classes = load_test_data(ds)

        for model_name in MODELS_TO_EVAL:
            print(f"  → Loading & Evaluating: {model_name}")
            try:
                if model_name in ["DNN", "CNN", "LSTM"]:
                    model_path = MODEL_DIR / f"{ds['name']}_{model_name}.h5"
                    if not model_path.exists():
                        print(f"     File not found: {model_path}")
                        continue
                    model = load_model(model_path)

                    if model_name in ["CNN", "LSTM"]:
                        X_test_input = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                        y_test_input = to_categorical(y_test) if model_name in ["CNN", "LSTM"] else y_test
                    else:
                        X_test_input = X_test
                        y_test_input = y_test

                    y_pred_prob = model.predict(X_test_input, verbose=0)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    y_score = y_pred_prob

                else:  # AdaBoost
                    model_path = MODEL_DIR / f"{ds['name']}_{model_name}.pkl"
                    if not model_path.exists():
                        print(f"     File not found: {model_path}")
                        continue
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    y_pred = model.predict(X_test)
                    y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                # Tính metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                FAR, FAP, FAF1, DR, FPR = compute_ids_metrics(y_test, y_pred)

                cm = confusion_matrix(y_test, y_pred)
                plot_confusion_matrix(cm, ds["name"], model_name)
                if y_score is not None:
                    plot_roc(y_test, y_score, ds["name"], model_name)

                results.append({
                    "Dataset": ds["name"],
                    "Model": model_name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "DetectionRate": DR,
                    "FPR": FPR,
                    "FAF1": FAF1,
                    "FAR": FAR,
                    "FAP": FAP,
                })
                print(f"     Done! Accuracy: {acc:.4f}")

            except Exception as e:
                print(f"     Error with {model_name}: {str(e)}")

    # Lưu kết quả
    if results:
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        try:
            df.to_excel(RESULTS_FILE, index=False)
            print(f"\nKết quả đánh giá đã lưu: {RESULTS_FILE}")
        except ImportError:
            df.to_csv("results/evaluated_results.csv", index=False, encoding='utf-8-sig')
            print("\nLưu dưới dạng CSV vì thiếu openpyxl: results/evaluated_results.csv")
    else:
        print("\nKhông có model nào được đánh giá thành công.")


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()