# =====================================================
# XAI VIEWER PRO VERSION (FULL FEATURES) - UPDATED
# =====================================================
import numpy as np
import pickle
import shap
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ================= PATH =================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
XAI_DIR = Path("xai_comparison_final4")
XAI_DIR.mkdir(exist_ok=True)
(XAI_DIR / "global").mkdir(exist_ok=True)
(XAI_DIR / "local").mkdir(exist_ok=True)
(XAI_DIR / "comparison").mkdir(exist_ok=True)

# ================= DATASETS & MODELS =================
DATASETS = [
    {"name": "CIC17", "X_test": "preprocessed_cic17_X_test.npy", "Y_test": "preprocessed_cic17_y_test.npy"},
    {"name": "NSL-KDD", "X_test": "preprocessed_nsl-kdd_X_test.npy", "Y_test": "preprocessed_nsl-kdd_y_test.npy"},
    {"name": "SIGNAL2022", "X_test": "signal22_X_test.npy", "Y_test": "signal22_y_test.npy"},
]

MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

# ================= GLOBAL VARIABLES =================
model = None
predict_fn = None
explainer = None
lime_explainer = None
X_test = None
y_test = None
feature_names = None
needs_3d = False
current_data = {}


# =====================================================
# LOAD PIPELINE
# =====================================================
def load_pipeline(dataset_name: str, model_name: str):
    global model, predict_fn, explainer, lime_explainer
    global X_test, y_test, feature_names, needs_3d

    try:
        # Load dataset
        ds_info = next(ds for ds in DATASETS if ds["name"] == dataset_name)
        X_test = np.load(DATA_DIR / ds_info["X_test"])
        y_test = np.load(DATA_DIR / ds_info["Y_test"])

        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)

        feature_names = [f"F_{i}" for i in range(X_test.shape[1])]
        is_deep = model_name in ["DNN", "CNN", "LSTM"]
        needs_3d = model_name in ["CNN", "LSTM"]

        # Load model
        if is_deep:
            model_path = MODEL_DIR / f"{dataset_name}_{model_name}.h5"
            model = tf.keras.models.load_model(model_path)

            def predict_fn_local(X):
                if needs_3d:
                    if X.ndim == 2:
                        X = X.reshape((X.shape[0], 1, X.shape[1]))
                    elif X.ndim == 1:
                        X = X.reshape((1, 1, -1))
                p = model.predict(X, verbose=0)
                return p if p.shape[1] > 1 else np.hstack([1 - p, p])

            predict_fn = predict_fn_local
        else:
            model_path = MODEL_DIR / f"{dataset_name}_{model_name}.pkl"
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            predict_fn = model.predict_proba

        # SHAP Explainer
        if is_deep:
            background = X_test[:50]
            if needs_3d:
                background = background.reshape((background.shape[0], 1, background.shape[1]))
            explainer = shap.DeepExplainer(model, background)
        else:
            explainer = shap.TreeExplainer(model)

        # LIME Explainer
        lime_explainer = LimeTabularExplainer(
            X_test[:100],
            feature_names=feature_names,
            class_names=["Benign", "Attack"],
            mode="classification"
        )

        print(f"✓ Loaded {dataset_name} - {model_name} successfully")

    except Exception as e:
        print(f"✗ Error loading pipeline: {e}")
        raise


# =====================================================
# GUI SETUP
# =====================================================
root = tk.Tk()
root.title("XAI Viewer PRO - Full Version")
root.geometry("1300x900")

# Top frame
frame_top = tk.Frame(root)
frame_top.pack(pady=10, fill="x")

tk.Label(frame_top, text="Dataset:").pack(side=tk.LEFT, padx=5)
dataset_box = ttk.Combobox(frame_top, values=[ds["name"] for ds in DATASETS], width=12)
dataset_box.set("CIC17")
dataset_box.pack(side=tk.LEFT, padx=5)

tk.Label(frame_top, text="Model:").pack(side=tk.LEFT, padx=5)
model_box = ttk.Combobox(frame_top, values=MODELS, width=15)
model_box.set("RandomForest")
model_box.pack(side=tk.LEFT, padx=5)

tk.Label(frame_top, text="Filter:").pack(side=tk.LEFT, padx=5)
filter_box = ttk.Combobox(frame_top, values=["All", "Only Benign", "Only Attack", "Wrong Only"], width=15)
filter_box.set("All")
filter_box.pack(side=tk.LEFT, padx=5)

tk.Label(frame_top, text="Sample Index:").pack(side=tk.LEFT, padx=5)
entry_idx = tk.Entry(frame_top, width=12)
entry_idx.pack(side=tk.LEFT, padx=5)

# Result Text with Scrollbar
result_text = scrolledtext.ScrolledText(root, height=18, font=("Consolas", 10))
result_text.pack(fill="x", padx=10, pady=5)
result_text.tag_config("wrong", foreground="red", font=("Consolas", 10, "bold"))
result_text.tag_config("highlight", background="yellow")
result_text.tag_config("positive", foreground="red")
result_text.tag_config("negative", foreground="blue")

# Plot frame
frame_plot = tk.Frame(root)
frame_plot.pack(fill="both", expand=True, padx=10, pady=5)



# =====================================================
# FILTER LOGIC
# =====================================================
def get_filtered_indices(filter_type: str):
    preds = predict_fn(X_test)
    preds = np.argmax(preds, axis=1)

    if filter_type == "Only Benign":
        return np.where(y_test == 0)[0]
    elif filter_type == "Only Attack":
        return np.where(y_test == 1)[0]
    elif filter_type == "Wrong Only":
        return np.where(preds != y_test)[0]
    else:
        return np.arange(len(X_test))


# =====================================================
# EXPLAIN SAMPLE
# =====================================================
def explain_sample():
    global canvas, current_data

    result_text.delete(1.0, tk.END)
    DATASET = dataset_box.get()
    MODEL = model_box.get()
    FILTER = filter_box.get()

    try:
        load_pipeline(DATASET, MODEL)
    except Exception as e:
        result_text.insert(tk.END, f"Lỗi load model/dataset: {e}\n", "wrong")
        return

    indices = get_filtered_indices(FILTER)

    try:
        idx = int(entry_idx.get())
        if idx < 0 or idx >= len(indices):
            raise ValueError
        real_idx = indices[idx]
    except:
        result_text.insert(tk.END, "Index không hợp lệ hoặc ngoài phạm vi!\n", "wrong")
        return

    x = X_test[real_idx]
    pred_prob = predict_fn(x.reshape(1, -1) if not needs_3d else x.reshape(1, 1, -1))
    pred = np.argmax(pred_prob)
    true = y_test[real_idx]

    result_text.insert(tk.END, f"Sample Index (filtered): {idx} | Real Index: {real_idx}\n")
    result_text.insert(tk.END, f"True Label : {true} ({'Benign' if true == 0 else 'Attack'})\n")
    result_text.insert(tk.END, f"Predicted   : {pred} ({'Benign' if pred == 0 else 'Attack'})\n")

    if pred != true:
        result_text.insert(tk.END, "⚠️  MISCLASSIFIED SAMPLE ⚠️\n", "wrong")

    # SHAP Explanation
    # ===== SHAP Explanation - ĐÃ SỬA =====
    try:
        if needs_3d:
            x_input = x.reshape(1, 1, -1)
        else:
            x_input = x.reshape(1, -1)

        sv = explainer.shap_values(x_input)

        # Xử lý output của SHAP cho DeepExplainer và TreeExplainer
        if isinstance(sv, list):
            # DeepExplainer thường trả về list [class0, class1]
            shap_vals = sv[1] if len(sv) > 1 else sv[0]  # Lấy class Attack (1)
        else:
            # TreeExplainer hoặc một số trường hợp khác
            shap_vals = sv[0] if sv.ndim == 3 else sv

        # Đảm bảo shap_vals là 1D (cho 1 sample)
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]  # Lấy sample đầu tiên

        # Top 10 features
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        shap_list = []

        result_text.insert(tk.END, "\n--- SHAP Top 10 Features ---\n")
        for i in top_idx:
            val = shap_vals[i]
            fname = feature_names[i]
            shap_list.append((fname, float(val)))
            tag = "positive" if val > 0 else "negative"
            result_text.insert(tk.END, f"{fname:>12}: {val:>8.4f}\n", tag)

    except Exception as e:
        result_text.insert(tk.END, f"Lỗi SHAP: {e}\n", "wrong")
        shap_list = []
        print(f"SHAP debug error: {e}")

    # LIME Explanation
    try:
        exp = lime_explainer.explain_instance(x, predict_fn, num_features=10)
        lime_list = exp.as_list()

        result_text.insert(tk.END, "\n--- LIME Top 10 Features ---\n")
        for f, v in lime_list:
            result_text.insert(tk.END, f"{f}: {v:.4f}\n")
    except Exception as e:
        result_text.insert(tk.END, f"Lỗi LIME: {e}\n", "wrong")

    # Save current data for export
    current_data = {
        "true": true,
        "pred": pred,
        "shap": shap_list,
        "lime": lime_list,
        "sample_idx": real_idx
    }

    top_features = [feature_names[i] for i in top_idx]
    top_values = shap_vals[top_idx]

    plt.barh(top_features[::-1], top_values[::-1])  # đảo ngược để feature quan trọng nhất ở trên
    plt.title(f"SHAP Local Explanation - {DATASET} | {MODEL} | Sample {real_idx}")
    plt.xlabel("SHAP Value")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# =====================================================
# EXPORT TO EXCEL
# =====================================================
def export_excel():
    if not current_data:
        tk.messagebox.showwarning("Warning", "Chưa có dữ liệu để export!")
        return

    file = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        initialdir=str(XAI_DIR / "local")
    )
    if not file:
        return

    try:
        df_shap = pd.DataFrame(current_data["shap"], columns=["Feature", "SHAP_Value"])
        df_lime = pd.DataFrame(current_data["lime"], columns=["Feature", "LIME_Value"])

        with pd.ExcelWriter(file) as writer:
            df_shap.to_excel(writer, sheet_name="SHAP", index=False)
            df_lime.to_excel(writer, sheet_name="LIME", index=False)

        tk.messagebox.showinfo("Success", f"Đã export thành công:\n{file}")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Export thất bại: {e}")


# =====================================================
# GLOBAL IMPORTANCE
# =====================================================
def show_global():
    try:
        sv = explainer.shap_values(X_test[:200])
        sv = sv[1] if isinstance(sv, list) else sv
        mean_abs = np.mean(np.abs(sv), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:15]

        plt.figure(figsize=(12, 8))
        plt.barh([feature_names[i] for i in top_idx[::-1]], mean_abs[top_idx][::-1])
        plt.title(f"Global Feature Importance - {dataset_box.get()} | {model_box.get()}")
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        tk.messagebox.showerror("Error", f"Lỗi Global Importance: {e}")


# =====================================================
# BUTTONS
# =====================================================
btn_frame = tk.Frame(root)
btn_frame.pack(pady=8)

tk.Button(btn_frame, text="Explain Sample", command=explain_sample, width=15, bg="#4CAF50", fg="white").pack(
    side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="Export Excel", command=export_excel, width=15, bg="#2196F3", fg="white").pack(side=tk.LEFT,
                                                                                                         padx=10)
tk.Button(btn_frame, text="Global Importance", command=show_global, width=18, bg="#FF9800", fg="white").pack(
    side=tk.LEFT, padx=10)

# Run GUI
root.mainloop()