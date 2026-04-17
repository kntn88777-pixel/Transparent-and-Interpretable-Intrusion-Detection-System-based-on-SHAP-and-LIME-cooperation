# =====================================================
# LOCAL SHAP ONLY - ĐÃ SỬA LỖI DeepSHAP (DNN, CNN, LSTM)
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
XAI_DIR = Path("xai_local_only_fixed")
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

print("=== LOCAL SHAP ONLY - ĐÃ SỬA DeepSHAP ===\n")

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
                background = X_input[:50] if len(X_input) > 50 else X_input
                explainer = shap.GradientExplainer(model, background)
                explainer_type = "DeepSHAP"
            else:
                background = X_input[:50] if len(X_input) > 50 else X_input
                explainer = shap.KernelExplainer(
                    model.predict if is_deep else model.predict_proba,
                    background
                )
                explainer_type = "KernelSHAP"

            # ==================== LOCAL SHAP ====================
            print(f" → Tính Local SHAP cho {N_LOCAL} mẫu ({explainer_type})...")

            X_to_explain = X_input[sample_indices[:N_LOCAL]]
            shap_values = explainer.shap_values(X_to_explain)

            # Xử lý shap_values và base_value
            if isinstance(shap_values, list):           # TreeSHAP / một số trường hợp
                sv_local = shap_values[1]               # class 1 (Attack)
                if hasattr(explainer, 'expected_value') and explainer.expected_value is not None:
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    base_value = np.mean(model.predict(X_input[:100]) if is_deep else model.predict_proba(X_input[:100])[:, 1])
            else:
                sv_local = shap_values
                if hasattr(explainer, 'expected_value') and explainer.expected_value is not None:
                    base_value = explainer.expected_value
                else:
                    # Tính base_value từ background (an toàn cho GradientExplainer)
                    preds = model.predict(background) if is_deep else model.predict_proba(background)
                    base_value = np.mean(preds) if preds.ndim == 1 else np.mean(preds[:, 1])

            # Vẽ waterfall cho từng mẫu
            for i, idx in enumerate(sample_indices[:N_LOCAL]):
                sv = sv_local[i] if len(sv_local.shape) > 1 else sv_local[i]

                explanation = shap.Explanation(
                    values=sv,
                    base_values=base_value,
                    data=X_input[idx],
                    feature_names=feature_names
                )

                plt.figure(figsize=(12, 8))
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

print("\n" + "="*90)
print("HOÀN TẤT - LOCAL SHAP ĐÃ SỬA DeepSHAP!")
print("="*90)
print(f"Kết quả lưu tại: {XAI_DIR / 'local'}")