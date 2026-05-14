# =====================================================
# XAI LOCAL GUI - SHAP + LIME (FINAL + HIGHLIGHT)
# =====================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer

# ================= PATH =================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")

DATASETS = [
    {"name": "CIC17", "X_test": "preprocessed_cic17_X_test.npy", "Y_test": "preprocessed_cic17_y_test.npy"},
    {"name": "NSL-KDD", "X_test": "preprocessed_nsl-kdd_X_test.npy", "Y_test": "preprocessed_nsl-kdd_y_test.npy"},
    {"name": "SIGNAL2022", "X_test": "preprocessed_signal22_X_test.npy", "Y_test": "preprocessed_signal22_y_test.npy"},
]

MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

# ================= GUI =================
root = tk.Tk()
root.title("XAI LOCAL VIEWER PRO")
root.geometry("1200x850")

frame_top = tk.Frame(root)
frame_top.pack(fill="x", padx=10, pady=10)

frame_plot = tk.Frame(root)
frame_plot.pack(fill="both", expand=True, padx=10, pady=10)

result_text = scrolledtext.ScrolledText(root, height=15, font=("Consolas", 10))
result_text.pack(fill="x", padx=10, pady=5)

# 🔥 HIGHLIGHT COLORS
result_text.tag_config("positive", foreground="red")
result_text.tag_config("negative", foreground="blue")
result_text.tag_config("highlight", background="yellow")

dataset_var = tk.StringVar(value="CIC17")
model_var = tk.StringVar(value="RandomForest")

# ================= UI =================
ttk.Label(frame_top, text="Dataset").grid(row=0, column=0, padx=5)
ttk.Combobox(frame_top, textvariable=dataset_var,
             values=[d["name"] for d in DATASETS], width=15).grid(row=0, column=1)

ttk.Label(frame_top, text="Model").grid(row=0, column=2, padx=5)
ttk.Combobox(frame_top, textvariable=model_var,
             values=MODELS, width=15).grid(row=0, column=3)

ttk.Label(frame_top, text="Sample Index").grid(row=0, column=4, padx=5)
entry_idx = tk.Entry(frame_top, width=10)
entry_idx.insert(0, "0")
entry_idx.grid(row=0, column=5)

# ================= LOAD =================
def load_dataset(name):
    ds = next(d for d in DATASETS if d["name"] == name)
    X = np.load(DATA_DIR / ds["X_test"])
    y = np.load(DATA_DIR / ds["Y_test"])
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    return X, y

def load_model(dataset, model_name):
    if model_name in ["DNN", "CNN", "LSTM"]:
        return tf.keras.models.load_model(MODEL_DIR / f"{dataset}_{model_name}.h5")
    else:
        with open(MODEL_DIR / f"{dataset}_{model_name}.pkl", "rb") as f:
            return pickle.load(f)

# 🔥 LOAD FEATURE NAMES
def load_feature_names(X, dataset_name):
    try:
        file_map = {
            "CIC17": "preprocessed_cic17_feature_names.pkl",
            "NSL-KDD": "preprocessed_nsl-kdd_feature_names.pkl",
            "SIGNAL2022": "preprocessed_signal22_feature_names.pkl"
        }

        file_path = DATA_DIR / file_map.get(dataset_name, "")

        if file_path.exists() and file_path.stat().st_size > 0:
            with open(file_path, "rb") as f:
                fn = pickle.load(f)

            if len(fn) == X.shape[1]:
                print(f"✓ Loaded {file_path.name}")
                return fn

    except Exception as e:
        print(f"⚠ Feature error: {e}")

    return [f"Feature_{i}" for i in range(X.shape[1])]

# ================= RESET =================
def reset_plot():
    for widget in frame_plot.winfo_children():
        widget.destroy()

# ================= RUN =================
def run_local():
    reset_plot()
    result_text.delete(1.0, tk.END)

    dataset_name = dataset_var.get()
    model_name = model_var.get()

    X_test, y_test = load_dataset(dataset_name)
    feature_names = load_feature_names(X_test, dataset_name)

    model = load_model(dataset_name, model_name)

    idx = int(entry_idx.get())
    x = X_test[idx]

    is_deep = model_name in ["DNN", "CNN", "LSTM"]
    needs_3d = model_name in ["CNN", "LSTM"]

    x_input = x.reshape(1, 1, -1) if needs_3d else x.reshape(1, -1)

    # ===== PREDICT =====
    pred_prob = model.predict(x_input, verbose=0) if is_deep else model.predict_proba(x_input)
    pred = np.argmax(pred_prob)
    true = y_test[idx]

    result_text.insert(tk.END, f"Sample: {idx}\nTrue: {true} | Pred: {pred}\n\n")

    # ===== SHAP =====
    if is_deep:
        background = X_test[:50]
        if needs_3d:
            background = background.reshape((background.shape[0], 1, background.shape[1]))
        explainer = shap.DeepExplainer(model, background)
        sv = explainer.shap_values(x_input)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x_input)

    shap_vals = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv
    shap_vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
    top1 = top_idx[0]

    result_text.insert(tk.END, "=== SHAP TOP FEATURES ===\n")

    for i in top_idx:
        val = shap_vals[i]
        fname = feature_names[i]

        tag = "positive" if val > 0 else "negative"

        if i == top1:
            result_text.insert(
                tk.END,
                f"{fname}: {val:.4f}  <-- MOST IMPORTANT\n",
                ("highlight", tag)
            )
        else:
            result_text.insert(tk.END, f"{fname}: {val:.4f}\n", tag)

    # ===== LIME =====
    lime_exp = LimeTabularExplainer(
        X_test[:100],
        feature_names=feature_names,
        class_names=["Benign", "Attack"],
        mode="classification"
    )

    def predict_fn(data):
        if needs_3d:
            data = data.reshape((data.shape[0], 1, data.shape[1]))
            return model.predict(data)
        return model.predict(data) if is_deep else model.predict_proba(data)

    exp = lime_exp.explain_instance(x, predict_fn, num_features=10)

    result_text.insert(tk.END, "\n=== LIME ===\n")
    for f, v in exp.as_list():
        result_text.insert(tk.END, f"{f}: {v:.4f}\n")

    # ===== PLOT =====
    colors = ["red" if v > 0 else "blue" for v in shap_vals[top_idx]]

    fig = plt.figure(figsize=(8, 5))
    plt.barh(
        [feature_names[i] for i in top_idx[::-1]],
        shap_vals[top_idx][::-1],
        color=colors[::-1]
    )

    plt.title(f"Local SHAP - Sample {idx}")
    plt.xlabel("SHAP Value")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ================= BUTTON =================
ttk.Button(frame_top, text="Explain", command=run_local).grid(row=1, column=2, pady=10)

root.mainloop()
