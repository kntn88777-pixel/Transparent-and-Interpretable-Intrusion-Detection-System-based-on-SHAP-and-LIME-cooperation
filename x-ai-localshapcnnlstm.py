# =====================================================
# LOCAL SHAP ONLY - ĐÃ SỬA HOÀN TOÀN CHO CNN & LSTM
# =====================================================
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import tensorflow as tf
import time
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ====================== PATHS & CONFIG ======================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
XAI_DIR = Path("xai_local_shap")
XAI_DIR.mkdir(exist_ok=True)
(XAI_DIR / "local").mkdir(exist_ok=True)

DATASETS = [
    {"name": "CIC17",     "X_test": "preprocessed_cic17_X_test.npy",     "Y_test": "preprocessed_cic17_y_test.npy"},
    {"name": "NSL-KDD",   "X_test": "preprocessed_nsl-kdd_X_test.npy",   "Y_test": "preprocessed_nsl-kdd_y_test.npy"},
    {"name": "SIGNAL2022","X_test": "signal22_X_test.npy",               "Y_test": "signal22_y_test.npy"},
]

MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

N_LOCAL = 5
sample_indices = [0, 1, 10, 50, 100]

print("=== LOCAL SHAP ONLY - FIXED CNN & LSTM ===\n")

for ds in DATASETS:
    dataset_name = ds["name"]
    print(f"\n{'#'*85}")
    print(f"DATASET: {dataset_name}")
    print(f"{'#'*85}")

    X_test = np.load(DATA_DIR / ds["X_test"])
    y_test = np.load(DATA_DIR / ds["Y_test"])
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    for model_name in MODELS:
        print(f"\n→ {model_name} trên {dataset_name}")
        start_time = time.time()

        try:
            # ==================== LOAD MODEL ====================
            if model_name in ["DNN", "CNN", "LSTM"]:
                model = tf.keras.models.load_model(MODEL_DIR / f"{dataset_name}_{model_name}.h5")
                if model_name in ["CNN", "LSTM"]:
                    X_input = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                else:
                    X_input = X_test
                is_deep = True
            else:
                with open(MODEL_DIR / f"{dataset_name}_{model_name}.pkl", "rb") as f:
                    model = pickle.load(f)
                X_input = X_test
                is_deep = False

            # ==================== CHỌN EXPLAINER ====================
            if model_name in ["RandomForest", "LightGBM"]:
                explainer = shap.TreeExplainer(model)
                explainer_type = "TreeSHAP"
            elif is_deep:
                background = X_input[:30] if len(X_input) > 30 else X_input  # Giảm background cho CNN/LSTM nhanh hơn
                explainer = shap.GradientExplainer(model, background)
                explainer_type = "DeepSHAP"
            else:
                background = X_input[:50] if len(X_input) > 50 else X_input
                explainer = shap.KernelExplainer(
                    model.predict if is_deep else model.predict_proba,
                    background
                )
                explainer_type = "KernelSHAP"

            # ==================== TÍNH SHAP VALUES ====================
            print(f" → Tính Local SHAP cho {N_LOCAL} mẫu ({explainer_type})...")

            X_to_explain = X_input[sample_indices[:N_LOCAL]]
            shap_values_raw = explainer.shap_values(X_to_explain)

            # ===================== XỬ LÝ SHAP VALUES & BASE VALUE =====================
            # GradientExplainer thường trả về list hoặc array có shape (samples, outputs, features)
            if isinstance(shap_values_raw, list):
                sv = shap_values_raw[1]          # Lấy class 1 (Attack)
            else:
                sv = shap_values_raw

            # Ép về shape (n_samples, n_features) - quan trọng cho CNN/LSTM
            if sv.ndim == 3:                     # shape (n_samples, 1, n_features)
                sv_local = sv[:, 0, :]           # Lấy output thứ nhất
            else:
                sv_local = sv

            # Tính base_value an toàn
            background_preds = model.predict(background, verbose=0)
            if background_preds.ndim == 2 and background_preds.shape[1] > 1:
                base_value = np.mean(background_preds[:, 1])   # probability của class 1
            else:
                base_value = np.mean(background_preds)         # sigmoid output

            # ==================== VẼ WATERFALL ====================
            for i, idx in enumerate(sample_indices[:N_LOCAL]):
                explanation = shap.Explanation(
                    values=sv_local[i],
                    base_values=base_value,
                    data=X_input[idx].flatten() if model_name in ["CNN", "LSTM"] else X_input[idx],
                    feature_names=feature_names
                )

                plt.figure(figsize=(13, 9))
                shap.plots.waterfall(explanation, show=False)
                plt.title(f"Local SHAP - Sample {idx} - {model_name} ({explainer_type}) - {dataset_name}")
                plt.tight_layout()

                save_path = XAI_DIR / "local" / f"{dataset_name}_{model_name}_local_sample{idx}.png"
                plt.savefig(save_path, dpi=280, bbox_inches='tight')
                plt.close()

                print(f"   ✓ Đã lưu sample {idx}")

            duration = round(time.time() - start_time, 1)
            print(f" ✓ Hoàn thành {model_name} trong {duration}s\n")

        except Exception as e:
            print(f" ✗ Lỗi với {model_name}: {str(e)}\n")

print("\n" + "="*95)
print("HOÀN TẤT - LOCAL SHAP ĐÃ FIX CNN & LSTM!")
print("="*95)
print(f"Kết quả lưu tại thư mục: {XAI_DIR / 'local'}")