# =====================================================
# XAI LOCAL GUI - SHAP + LIME (FINAL + AUTO PICK + MAP)
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
from sklearn.decomposition import PCA
import pandas as pd

# ================= PATH =================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
RAW_FEATURE_FILE = r"D:\luanvsn\.venv\final\data\CICIDS2017.csv"

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

# highlight
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

# ================= FEATURE NAMES =================
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
                return fn

        if dataset_name == "CIC17" and Path(RAW_FEATURE_FILE).exists():
            df = pd.read_csv(RAW_FEATURE_FILE, nrows=1)
            cols = [c for c in df.columns if c.lower() not in ["label", "class"]]
            return cols[:X.shape[1]]

    except:
        pass

    return [f"Feature_{i}" for i in range(X.shape[1])]

# ================= AUTO PICK =================
def auto_pick_sample(X_test, y_test, model, is_deep, needs_3d):
    X_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1])) if needs_3d else X_test
    pred_prob = model.predict(X_input, verbose=0) if is_deep else model.predict_proba(X_input)
    preds = np.argmax(pred_prob, axis=1)

    wrong_idx = np.where(preds != y_test)[0]
    if len(wrong_idx) > 0:
        return int(wrong_idx[0])

    confidence = np.max(pred_prob, axis=1)
    return int(np.argsort(confidence)[0])

# ================= RESET =================
def reset_plot():
    for widget in frame_plot.winfo_children():
        widget.destroy()

# ================= LOCAL =================
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

    pred_prob = model.predict(x_input, verbose=0) if is_deep else model.predict_proba(x_input)
    pred = np.argmax(pred_prob)
    true = y_test[idx]

    result_text.insert(tk.END, f"Sample: {idx}\nTrue: {true} | Pred: {pred}\n\n")

    # SHAP
    if is_deep:
        background = X_test[:50]
        if needs_3d:
            background = background.reshape((background.shape[0], 1, background.shape[1]))
        explainer = shap.DeepExplainer(model, background)
        sv = explainer.shap_values(x_input)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x_input)

    shap_vals = sv[1] if isinstance(sv, list) else sv
    shap_vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
    top1 = top_idx[0]

    for i in top_idx:
        val = shap_vals[i]
        fname = feature_names[i]
        tag = "positive" if val > 0 else "negative"

        if i == top1:
            result_text.insert(tk.END,
                f"{fname}: {val:.4f} <-- MOST IMPORTANT\n",
                ("highlight", tag))
        else:
            result_text.insert(tk.END,
                f"{fname}: {val:.4f}\n",
                tag)

    # plot
    fig = plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in top_idx[::-1]],
             shap_vals[top_idx][::-1])
    plt.title(f"Local SHAP - Sample {idx}")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ================= MAP =================
def show_map():
    reset_plot()

    dataset_name = dataset_var.get()
    model_name = model_var.get()

    X_test, y_test = load_dataset(dataset_name)
    model = load_model(dataset_name, model_name)

    is_deep = model_name in ["DNN", "CNN", "LSTM"]
    needs_3d = model_name in ["CNN", "LSTM"]

    X_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1])) if needs_3d else X_test
    pred_prob = model.predict(X_input, verbose=0) if is_deep else model.predict_proba(X_input)

    preds = np.argmax(pred_prob, axis=1)
    confidence = np.max(pred_prob, axis=1)

    X_2d = PCA(n_components=2).fit_transform(X_test[:1000])

    colors = []
    for i in range(1000):
        if preds[i] != y_test[i]:
            colors.append("yellow")
        elif confidence[i] < 0.6:
            colors.append("orange")
        elif preds[i] == 1:
            colors.append("red")
        else:
            colors.append("blue")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=20)

    def on_click(event):
        if event.inaxes != ax:
            return
        dist = np.sqrt((X_2d[:,0]-event.xdata)**2 + (X_2d[:,1]-event.ydata)**2)
        idx = np.argmin(dist)

        entry_idx.delete(0, tk.END)
        entry_idx.insert(0, str(idx))
        run_local()

    fig.canvas.mpl_connect('button_press_event', on_click)

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ================= BUTTON =================
ttk.Button(frame_top, text="Explain", command=run_local).grid(row=1, column=2)
ttk.Button(frame_top, text="Auto Pick", command=lambda: entry_idx.insert(0, "0")).grid(row=1, column=3)
ttk.Button(frame_top, text="Show Map", command=show_map).grid(row=1, column=4)

root.mainloop()
