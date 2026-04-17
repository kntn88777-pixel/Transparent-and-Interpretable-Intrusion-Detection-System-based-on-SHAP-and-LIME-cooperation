# =====================================================
# XAI GLOBAL GUI - FIX SIZE + RESET BUTTON
# =====================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tkinter as tk
from tkinter import ttk
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from pathlib import Path

# ================= PATH =================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")

DATASETS = [
    {"name": "CIC17", "X_test": "preprocessed_cic17_X_test.npy", "Y_test": "preprocessed_cic17_y_test.npy"},
    {"name": "NSL-KDD", "X_test": "preprocessed_nsl-kdd_X_test.npy", "Y_test": "preprocessed_nsl-kdd_y_test.npy"},
    {"name": "SIGNAL2022", "X_test": "signal22_X_test.npy", "Y_test": "signal22_y_test.npy"},
]

MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

N_GLOBAL = 80

# ================= GUI =================
root = tk.Tk()
root.title("XAI GLOBAL VIEWER PRO")
root.geometry("1200x800")

# ===== Frame điều khiển =====
frame_control = tk.Frame(root, height=150)
frame_control.pack(fill="x", padx=10, pady=10)

# ===== Frame plot =====
frame_plot = tk.Frame(root)
frame_plot.pack(fill="both", expand=True, padx=10, pady=10)

dataset_var = tk.StringVar(value="NSL-KDD")
model_var = tk.StringVar(value="RandomForest")
explainer_var = tk.StringVar(value="TreeSHAP")

# Dropdowns
ttk.Label(frame_control, text="Dataset").grid(row=0, column=0, padx=5)
ttk.Combobox(frame_control, textvariable=dataset_var,
             values=[d["name"] for d in DATASETS], width=20).grid(row=0, column=1)

ttk.Label(frame_control, text="Model").grid(row=0, column=2, padx=5)
ttk.Combobox(frame_control, textvariable=model_var,
             values=MODELS, width=20).grid(row=0, column=3)

ttk.Label(frame_control, text="Explainer").grid(row=0, column=4, padx=5)
ttk.Combobox(frame_control, textvariable=explainer_var,
             values=["TreeSHAP", "KernelSHAP", "DeepSHAP", "LIME"], width=20).grid(row=0, column=5)

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

# ================= RESET =================
def reset_plot():
    for widget in frame_plot.winfo_children():
        widget.destroy()

# ================= RUN =================
def run_xai():
    reset_plot()

    dataset_name = dataset_var.get()
    model_name = model_var.get()
    explainer_type = explainer_var.get()

    X_test, y_test = load_dataset(dataset_name)
    feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
    X = X_test[:N_GLOBAL]

    model = load_model(dataset_name, model_name)

    is_deep = model_name in ["DNN", "CNN", "LSTM"]
    needs_3d = model_name in ["CNN", "LSTM"]

    if needs_3d:
        X_3d = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        X_3d = X

    # ===== SHAP =====
    if explainer_type == "TreeSHAP":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv

    elif explainer_type == "KernelSHAP":
        explainer = shap.KernelExplainer(model.predict_proba, X[:30])
        sv = explainer.shap_values(X)[1]

    elif explainer_type == "DeepSHAP":
        if model_name == "LSTM":
            print("Skip LSTM DeepSHAP")
            return

        explainer = shap.DeepExplainer(model, X_3d[:50])
        sv = explainer.shap_values(X_3d)

        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]

        if sv.ndim == 3:
            sv = sv[:, 0, :]

    # ===== LIME =====
    elif explainer_type == "LIME":
        from lime.lime_tabular import LimeTabularExplainer

        lime_exp = LimeTabularExplainer(
            X,
            feature_names=feature_names,
            class_names=["Benign", "Attack"],
            mode="classification"
        )

        lime_values = np.zeros((len(X), X.shape[1]))

        def predict_fn(data):
            if needs_3d:
                data = data.reshape((data.shape[0], 1, data.shape[1]))
                return model.predict(data)
            return model.predict(data) if is_deep else model.predict_proba(data)

        for i in range(len(X)):
            exp = lime_exp.explain_instance(X[i], predict_fn, num_features=X.shape[1])
            for feat, val in exp.as_list():
                idx = int(feat.split("_")[1])
                lime_values[i, idx] = val

        sv = lime_values

    # ===== PLOT (FIX SIZE) =====
    fig = plt.figure(figsize=(9, 5.5))  # 👈 FIX SIZE

    shap.summary_plot(
        sv,
        X,
        feature_names=feature_names,
        show=False
    )

    plt.title(f"{explainer_type} - {model_name} - {dataset_name}", fontsize=11)

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ===== BUTTONS =====
ttk.Button(frame_control, text="Run", command=run_xai).grid(row=1, column=2, pady=10)
ttk.Button(frame_control, text="Reset", command=reset_plot).grid(row=1, column=3, pady=10)

root.mainloop()