# =====================================================
# XAI METRICS FULL - FINAL (ADD LSTM + FIX ALL BUGS)
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
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ================= PATH =================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
SAVE_DIR = Path("metric1")
SAVE_DIR.mkdir(exist_ok=True)

DATASETS = [
    {"name": "CIC17", "X_test": "preprocessed_cic17_X_test.npy", "Y_test": "preprocessed_cic17_y_test.npy"},
    {"name": "SIGNAL2022", "X_test": "signal22_X_test.npy", "Y_test": "signal22_y_test.npy"},
    {"name": "NSL-KDD",
     "X_test": "preprocessed_nsl-kdd_X_test.npy","Y_test": "preprocessed_nsl-kdd_y_test.npy"}
]

# ✅ THÊM LSTM
MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

# ================= CONFIG =================
N_SAMPLES = 15
NOISE = 0.03
BACKGROUND_SAMPLES = 30

# =====================================================
# ================= METRICS =================
# =====================================================

def deletion_metric(model_fn, x, w, y_true, baseline):
    x = x.flatten()
    order = np.argsort(-np.abs(w))
    scores = []
    for i in range(len(w) + 1):
        x_mod = x.copy()
        x_mod[order[:i]] = baseline[order[:i]]
        pred = model_fn(x_mod.reshape(1, -1))[0]
        scores.append(pred[y_true])
    return np.mean(scores)


def insertion_metric(model_fn, x, w, y_true, baseline):
    x = x.flatten()
    order = np.argsort(-np.abs(w))
    scores = []
    for i in range(len(w) + 1):
        x_mod = baseline.copy()
        x_mod[order[:i]] = x[order[:i]]
        pred = model_fn(x_mod.reshape(1, -1))[0]
        scores.append(pred[y_true])
    return np.mean(scores)


def infidelity(model_fn, x, w, noise_std=0.02, n_samples=15):
    x = x.flatten()
    scores = []
    for _ in range(n_samples):
        noise = np.random.normal(0, noise_std, x.shape)
        x_noisy = x + noise
        pred_diff = model_fn(x.reshape(1, -1))[0].max() - model_fn(x_noisy.reshape(1, -1))[0].max()
        approx = np.dot(w, noise)
        scores.append((pred_diff - approx) ** 2)
    return np.mean(scores)


def importance_corr(model_fn, x, w, y_true):
    x = x.flatten()
    changes = []
    for i in range(len(w)):
        x_mod = x.copy()
        x_mod[i] = 0
        pred_orig = model_fn(x.reshape(1, -1))[0]
        pred_mod = model_fn(x_mod.reshape(1, -1))[0]
        changes.append(pred_orig[y_true] - pred_mod[y_true])
    corr, _ = spearmanr(w, changes)
    return corr if not np.isnan(corr) else 0.0


def robustness(w1, w2):
    return np.linalg.norm(w1 - w2)


def sparsity(w):
    return np.count_nonzero(np.abs(w) > 1e-8) / len(w)


def entropy(w):
    w = np.abs(w)
    w = w / (np.sum(w) + 1e-12)
    return -np.sum(w * np.log(w + 1e-12))


def topk_ratio(w, k=5):
    abs_w = np.abs(w)
    return np.sum(np.sort(abs_w)[-k:]) / (np.sum(abs_w) + 1e-12)


def consistency(w1, w2):
    sp = spearmanr(w1, w2)[0]
    cos = 1 - cosine(w1, w2)
    return sp if not np.isnan(sp) else 0.0, cos


# =====================================================
# ================= MAIN =================
# =====================================================
print("=== XAI FULL METRICS FINAL ===")

for ds in DATASETS:
    dataset_name = ds["name"]

    X_test = np.load(DATA_DIR / ds["X_test"])
    y_test = np.load(DATA_DIR / ds["Y_test"])
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    baseline = np.mean(X_test, axis=0)
    feature_names = [f"F{i}" for i in range(X_test.shape[1])]

    for model_name in MODELS:
        print(f"\n→ {model_name}")

        try:
            # ================= LOAD MODEL =================
            if model_name in ["DNN", "CNN", "LSTM"]:
                model = tf.keras.models.load_model(MODEL_DIR / f"{dataset_name}_{model_name}.h5")
                is_keras = True
                needs_3d = (model_name in ["CNN", "LSTM"])
            else:
                with open(MODEL_DIR / f"{dataset_name}_{model_name}.pkl", "rb") as f:
                    model = pickle.load(f)
                is_keras = False
                needs_3d = False

            # ================= PREDICT FN =================
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

            # ================= EXPLAINERS =================
            kernel_explainer = shap.KernelExplainer(predict_fn, X_test[:BACKGROUND_SAMPLES])
            tree_explainer = None
            deep_explainer = None

            if model_name in ["RandomForest", "LightGBM"]:
                tree_explainer = shap.TreeExplainer(model)

            # ❌ KHÔNG dùng DeepSHAP cho LSTM
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

            rows = []
            indices = np.random.choice(len(X_test), N_SAMPLES, replace=False)

            for idx in indices:
                x = X_test[idx:idx+1]
                y_true = y_test[idx]

                shap_dict = {}

                # ================= KernelSHAP =================
                sv = kernel_explainer.shap_values(x)
                sv = sv[1] if isinstance(sv, list) else sv
                shap_dict["KernelSHAP"] = sv.flatten()

                # ================= TreeSHAP =================
                if tree_explainer:
                    sv = tree_explainer.shap_values(x)
                    sv = sv[1] if isinstance(sv, list) else sv
                    shap_dict["TreeSHAP"] = sv.flatten()

                # ================= DeepSHAP =================
                if deep_explainer:
                    x_input = x
                    if model_name == "CNN":
                        x_input = x.reshape((1, 1, x.shape[1]))

                    sv = deep_explainer.shap_values(x_input)

                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]

                    if sv.ndim == 3:
                        sv = sv[:, 0, :]

                    shap_dict["DeepSHAP"] = sv.flatten()

                # ================= LIME =================
                exp = lime_exp.explain_instance(x[0], predict_fn, num_features=len(feature_names))
                lime_w = np.zeros(len(feature_names))

                class_id = list(exp.local_exp.keys())[0]
                for fid, weight in exp.local_exp[class_id]:
                    lime_w[fid] = weight

                shap_dict["LIME"] = lime_w

                row = {}

                # ================= METRICS =================
                for name, w in shap_dict.items():
                    p = name.lower().replace("shap", "")

                    row[f"del_{p}"] = deletion_metric(predict_fn, x[0], w, y_true, baseline)
                    row[f"ins_{p}"] = insertion_metric(predict_fn, x[0], w, y_true, baseline)
                    row[f"infid_{p}"] = infidelity(predict_fn, x[0], w)
                    row[f"corr_{p}"] = importance_corr(predict_fn, x[0], w, y_true)
                    row[f"sparsity_{p}"] = sparsity(w)
                    row[f"entropy_{p}"] = entropy(w)
                    row[f"topk_{p}"] = topk_ratio(w)

                # ================= ROBUSTNESS =================
                noise = np.random.normal(0, NOISE, x.shape)
                sv_noisy = kernel_explainer.shap_values(x + noise)
                sv_noisy = sv_noisy[1] if isinstance(sv_noisy, list) else sv_noisy
                row["robust"] = robustness(shap_dict["KernelSHAP"], sv_noisy.flatten())

                # ================= CONSISTENCY =================
                if "KernelSHAP" in shap_dict and "LIME" in shap_dict:
                    sp, cos = consistency(shap_dict["KernelSHAP"], shap_dict["LIME"])
                    row["spearman"] = sp
                    row["cosine"] = cos

                rows.append(row)

            df = pd.DataFrame(rows)
            m = df.mean(numeric_only=True)

            metrics_names = [
                "Deletion", "Insertion", "Infidelity", "Importance",
                "Robustness", "Sparsity", "Entropy", "TopK_Ratio",
                "Spearman", "Cosine"
            ]

            result_rows = []

            for metric in metrics_names:
                row = {"Model": model_name, "Metric": metric}

                mapping = {
                    "Deletion": "del_",
                    "Insertion": "ins_",
                    "Infidelity": "infid_",
                    "Importance": "corr_",
                    "Sparsity": "sparsity_",
                    "Entropy": "entropy_",
                    "TopK_Ratio": "topk_"
                }

                if metric in mapping:
                    key = mapping[metric]
                    row["KernelSHAP"] = m.get(key + "kernel", np.nan)
                    row["TreeSHAP"]   = m.get(key + "tree", np.nan)
                    row["DeepSHAP"]   = m.get(key + "deep", np.nan)
                    row["LIME"]       = m.get(key + "lime", np.nan)

                elif metric == "Robustness":
                    val = m.get("robust", np.nan)
                    row.update(dict.fromkeys(["KernelSHAP","TreeSHAP","DeepSHAP","LIME"], val))

                elif metric == "Spearman":
                    val = m.get("spearman", np.nan)
                    row.update(dict.fromkeys(["KernelSHAP","TreeSHAP","DeepSHAP","LIME"], val))

                elif metric == "Cosine":
                    val = m.get("cosine", np.nan)
                    row.update(dict.fromkeys(["KernelSHAP","TreeSHAP","DeepSHAP","LIME"], val))

                result_rows.append(row)

            result_df = pd.DataFrame(result_rows)

            save_file = SAVE_DIR / f"{dataset_name}_{model_name}_FINAL.xlsx"
            result_df.to_excel(save_file, index=False)

            print(f"✓ Saved: {save_file}")

        except Exception as e:
            print(f"❌ Error {model_name}: {e}")

print("\n=== DONE FULL ===")