# =====================================================
# XAI COMPARISON FINAL 2 - FULL FIX (LIME + METRIC)
# =====================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path
import tensorflow as tf
import time
import warnings
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ====================== PATHS ======================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
XAI_DIR = Path("xai9+."
               "")

XAI_DIR.mkdir(exist_ok=True)
(XAI_DIR / "global").mkdir(exist_ok=True)
(XAI_DIR / "local").mkdir(exist_ok=True)
(XAI_DIR / "comparison").mkdir(exist_ok=True)

DATASETS = [
    {"name": "CIC17", "X_test": "preprocessed_cic17_X_test.npy", "Y_test": "preprocessed_cic17_y_test.npy"},
    {"name": "NSL-KDD", "X_test": "preprocessed_nsl-kdd_X_test.npy", "Y_test": "preprocessed_nsl-kdd_y_test.npy"},
    {"name": "SIGNAL2022", "X_test": "signal22_X_test.npy", "Y_test": "signal22_y_test.npy"},
]

MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

N_GLOBAL = 80
N_LOCAL = 3
N_METRIC_SAMPLES = 20
N_PERTURB_FIDELITY = 150
NOISE_LEVEL_ROBUST = 0.03
N_RUNS_STABILITY = 5

comparison_table = []

print("=== XAI FINAL 2 + METRIC - FULL FIX ===\n")

for ds in DATASETS:
    dataset_name = ds["name"]
    print(f"\n{'#'*90}")
    print(f"DATASET: {dataset_name}")
    print(f"{'#'*90}")

    X_test = np.load(DATA_DIR / ds["X_test"])
    y_test = np.load(DATA_DIR / ds["Y_test"])
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    for model_name in MODELS:
        print(f"\n→ {model_name} trên {dataset_name}")
        start_time = time.time()

        try:
            is_deep = False
            needs_3d = False

            # ================= LOAD MODEL =================
            if model_name in ["DNN", "CNN", "LSTM"]:
                model = tf.keras.models.load_model(MODEL_DIR / f"{dataset_name}_{model_name}.h5")
                is_deep = True

                if model_name in ["CNN", "LSTM"]:
                    X_input_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                    needs_3d = True
                else:
                    X_input_3d = X_test

                X_input_2d = X_test

                if model_name == "LSTM":
                    print(" → Skip LSTM (SHAP bug)")
                    continue

                background = X_input_3d[:60]
                explainer = shap.DeepExplainer(model, background)
                explainer_type = "DeepSHAP"

            else:
                with open(MODEL_DIR / f"{dataset_name}_{model_name}.pkl", "rb") as f:
                    model = pickle.load(f)

                X_input_2d = X_test

                if model_name in ["RandomForest", "LightGBM"]:
                    explainer = shap.TreeExplainer(model)
                    explainer_type = "TreeSHAP"
                else:
                    explainer = shap.KernelExplainer(model.predict_proba, X_input_2d[:50])
                    explainer_type = "KernelSHAP"

            # ================= PREDICT WRAPPER =================
            def predict_wrapper(X):
                try:
                    if needs_3d:
                        X = X.reshape((X.shape[0], 1, X.shape[1]))

                    if not is_deep:
                        if hasattr(model, "predict_proba"):
                            return model.predict_proba(X)
                        else:
                            preds = model.predict(X).astype(int)
                            return np.eye(len(np.unique(y_test)))[preds]

                    preds = model.predict(X, verbose=0)

                    if preds.ndim == 2 and preds.shape[1] == 1:
                        preds = np.hstack([1 - preds, preds])

                    return preds

                except:
                    return np.zeros((X.shape[0], 2))

            # ================= GLOBAL SHAP =================
            try:
                print(f" → Global {explainer_type}")

                if explainer_type == "DeepSHAP":
                    sv = explainer.shap_values(X_input_3d[:N_GLOBAL])
                    sv = sv[1] if isinstance(sv, list) else sv
                    if sv.ndim == 3:
                        sv = sv[:, 0, :]
                else:
                    sv = explainer.shap_values(X_input_2d[:N_GLOBAL])
                    sv = sv[1] if isinstance(sv, list) else sv

                plt.figure(figsize=(12, 8))
                shap.summary_plot(sv, X_input_2d[:N_GLOBAL], feature_names=feature_names, show=False)
                plt.savefig(XAI_DIR / "global" / f"{dataset_name}_{model_name}.png")
                plt.close()

                global_done = True
            except Exception as e:
                print(" SHAP lỗi:", e)
                global_done = False

            # ================= LIME =================
            lime_success = False
            try:
                print(" → LIME")

                lime_explainer = LimeTabularExplainer(
                    X_input_2d[:100],
                    feature_names=feature_names,
                    class_names=["0", "1"],
                    mode="classification"
                )

                for idx in [0, 10]:
                    exp = lime_explainer.explain_instance(
                        X_input_2d[idx],
                        predict_wrapper,
                        num_features=15
                    )
                    exp.save_to_file(str(XAI_DIR / "local" / f"{dataset_name}_{model_name}_{idx}.html"))

                lime_success = True
            except Exception as e:
                print(" LIME lỗi:", e)

            # ================= METRIC =================
            if global_done or lime_success:
                print(" → Metric...")

                rows = []
                indices = np.random.choice(len(X_input_2d), min(N_METRIC_SAMPLES, len(X_input_2d)), replace=False)

                for idx in indices:
                    instance = X_input_2d[idx:idx+1]

                    probs = predict_wrapper(instance)
                    if probs.ndim == 1:
                        probs = probs.reshape(1, -1)

                    class_idx = int(np.argmax(probs[0]))

                    row = {
                        "Dataset": dataset_name,
                        "Model": model_name
                    }

                    # ===== Robustness =====
                    noise = np.random.normal(0, NOISE_LEVEL_ROBUST, instance.shape)
                    noisy = instance + noise

                    p1 = predict_wrapper(instance)[0]
                    p2 = predict_wrapper(noisy)[0]

                    row["Robustness"] = float(np.linalg.norm(p1 - p2))

                    # ===== Fidelity =====
                    pert = instance + np.random.normal(
                        0, 0.05 * np.std(X_input_2d, axis=0),
                        (N_PERTURB_FIDELITY, X_input_2d.shape[1])
                    )

                    preds_pert = predict_wrapper(pert)
                    if preds_pert.ndim == 1:
                        preds_pert = preds_pert.reshape(-1, 1)

                    true_preds = preds_pert[:, class_idx]

                    try:
                        shap_vals = explainer.shap_values(instance)
                        shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                        shap_vals = shap_vals.flatten()

                        base = explainer.expected_value
                        base = base[class_idx] if isinstance(base, (list, np.ndarray)) else base

                        shap_pred = base + np.dot(pert - instance, shap_vals)
                        row["Fidelity_SHAP"] = r2_score(true_preds, shap_pred)
                    except:
                        pass

                    rows.append(row)

                pd.DataFrame(rows).to_excel(
                    XAI_DIR / "comparison" / f"{dataset_name}_{model_name}.xlsx",
                    index=False
                )

            duration = round(time.time() - start_time, 2)

            comparison_table.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Explainer": explainer_type,
                "Time": duration
            })

            print(f" ✓ Done {duration}s")

        except Exception as e:
            print(" ✗ Lỗi:", e)

# ================= SAVE =================
df = pd.DataFrame(comparison_table)
df.to_excel(XAI_DIR / "comparison" / "summary.xlsx", index=False)

print("\nDONE ALL")