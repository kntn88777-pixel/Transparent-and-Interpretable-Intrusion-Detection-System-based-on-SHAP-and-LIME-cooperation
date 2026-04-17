# =====================================================
# DATASET COVERAGE (EXPLAINER RELIABILITY - FINAL)
# =====================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pickle
import shap
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ================= PATH =================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
SAVE_DIR = Path("coverage_dataset")
SAVE_DIR.mkdir(exist_ok=True)

DATASETS = [
    {"name": "CIC17", "X_test": "preprocessed_cic17_X_test.npy", "Y_test": "preprocessed_cic17_y_test.npy"},
    {"name": "SIGNAL2022", "X_test": "signal22_X_test.npy", "Y_test": "signal22_y_test.npy"},
    {"name": "NSL-KDD",
     "X_test": "preprocessed_nsl-kdd_X_test.npy","Y_test": "preprocessed_nsl-kdd_y_test.npy"}
]

MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

# ================= CONFIG =================
N_SAMPLES = 200  # tăng để gần "dataset-level"
BACKGROUND_SAMPLES = 30

print("=== DATASET COVERAGE ===")

for ds in DATASETS:
    dataset_name = ds["name"]

    X_test = np.load(DATA_DIR / ds["X_test"])
    y_test = np.load(DATA_DIR / ds["Y_test"])
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    feature_names = [f"F{i}" for i in range(X_test.shape[1])]

    for model_name in MODELS:
        print(f"\n→ {dataset_name} | {model_name}")

        try:
            # ===== LOAD MODEL =====
            if model_name in ["DNN", "CNN", "LSTM"]:
                model = tf.keras.models.load_model(MODEL_DIR / f"{dataset_name}_{model_name}.h5")
                is_keras = True
                needs_3d = (model_name in ["CNN", "LSTM"])
            else:
                with open(MODEL_DIR / f"{dataset_name}_{model_name}.pkl", "rb") as f:
                    model = pickle.load(f)
                is_keras = False
                needs_3d = False

            def predict_fn(X):
                if needs_3d:
                    X = X.reshape((X.shape[0], 1, X.shape[1]))

                if is_keras:
                    preds = model.predict(X, verbose=0)
                else:
                    preds = model.predict_proba(X)

                if preds.ndim == 1:
                    preds = np.vstack([1 - preds, preds]).T

                if preds.shape[1] == 1:
                    preds = np.hstack([1 - preds, preds])

                return preds

            # ===== EXPLAINERS =====
            kernel_explainer = shap.KernelExplainer(
                predict_fn, X_test[:BACKGROUND_SAMPLES]
            )

            tree_explainer = None
            if model_name in ["RandomForest", "LightGBM"]:
                tree_explainer = shap.TreeExplainer(model)

            deep_explainer = None
            if model_name in ["DNN", "CNN"]:
                try:
                    bg = X_test[:BACKGROUND_SAMPLES]
                    if model_name == "CNN":
                        bg = bg.reshape((bg.shape[0], 1, bg.shape[1]))
                    deep_explainer = shap.DeepExplainer(model, bg)
                except:
                    deep_explainer = None

            lime_exp = LimeTabularExplainer(
                X_test[:100],
                feature_names=feature_names,
                mode="classification"
            )

            # ===== SAMPLING =====
            N = min(N_SAMPLES, len(X_test))
            indices = np.random.choice(len(X_test), N, replace=False)

            total = 0
            success = {
                "KernelSHAP": 0,
                "TreeSHAP": 0,
                "DeepSHAP": 0,
                "LIME": 0
            }

            # ===== LOOP =====
            for idx in indices:
                x = X_test[idx:idx+1]
                total += 1

                # KernelSHAP
                try:
                    sv = kernel_explainer.shap_values(x)
                    if sv is not None:
                        success["KernelSHAP"] += 1
                except:
                    pass

                # TreeSHAP
                if tree_explainer:
                    try:
                        sv = tree_explainer.shap_values(x)
                        if sv is not None:
                            success["TreeSHAP"] += 1
                    except:
                        pass

                # DeepSHAP
                if deep_explainer:
                    try:
                        x_input = x
                        if model_name == "CNN":
                            x_input = x.reshape((1, 1, x.shape[1]))

                        sv = deep_explainer.shap_values(x_input)
                        if sv is not None:
                            success["DeepSHAP"] += 1
                    except:
                        pass

                # LIME
                try:
                    exp = lime_exp.explain_instance(x[0], predict_fn)
                    if exp is not None:
                        success["LIME"] += 1
                except:
                    pass

            # ===== RESULT =====
            result = {
                "Model": model_name,
                "Metric": "Coverage"
            }

            for k in success:
                result[k] = success[k] / total

            df_result = pd.DataFrame([result])

            save_file = SAVE_DIR / f"{dataset_name}_{model_name}_coverage.xlsx"
            df_result.to_excel(save_file, index=False)

            print(f"✓ Saved: {save_file}")

        except Exception as e:
            print(f"❌ Error {model_name}: {e}")

print("\n=== DONE DATASET COVERAGE ===")