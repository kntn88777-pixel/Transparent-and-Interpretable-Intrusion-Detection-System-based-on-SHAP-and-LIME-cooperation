# =====================================================
# XAI COMPARISON FINAL 2 - FINAL FIX LIME CHO CNN/LSTM (RESHAPE WRAPPER)
# =====================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Tắt GPU

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

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ====================== PATHS ======================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
XAI_DIR = Path("xai_comparison_final4")
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

comparison_table = []

print("=== XAI FINAL 2 - ĐÃ FIX LIME CHO CNN/LSTM BẰNG RESHAPE WRAPPER ===\n")

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
        global_done = False
        lime_success = False
        explainer_type = "None"

        try:
            is_deep = False
            needs_3d = False

            # Load model
            if model_name in ["DNN", "CNN", "LSTM"]:
                model = tf.keras.models.load_model(MODEL_DIR / f"{dataset_name}_{model_name}.h5")
                print(f"Debug: Model input shape = {model.input_shape}")
                is_deep = True

                if model_name in ["CNN", "LSTM"]:
                    X_input_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                    X_input_2d = X_test
                    needs_3d = True
                else:
                    X_input_3d = X_test
                    X_input_2d = X_test

                if model_name == "LSTM":
                    explainer_type = "Skipped (LSTM bug)"
                    print(" → Bỏ SHAP cho LSTM (bug thư viện)")
                else:
                    N_BACKGROUND = 60
                    background_idx = np.random.choice(len(X_input_3d), size=min(N_BACKGROUND, len(X_input_3d)), replace=False)
                    background_data = X_input_3d[background_idx]

                    explainer = shap.DeepExplainer(model, background_data)
                    explainer_type = "DeepExplainer"

                    print(f" → Global {explainer_type} (N={N_GLOBAL}) ...")
                    try:
                        sv_global_raw = explainer.shap_values(X_input_3d[:N_GLOBAL])
                        if isinstance(sv_global_raw, list):
                            sv_global = sv_global_raw[1] if len(sv_global_raw) > 1 else sv_global_raw[0]
                        else:
                            sv_global = sv_global_raw

                        if sv_global.ndim == 3 and sv_global.shape[1] == 1:
                            sv_global = sv_global[:, 0, :]

                        print(f"   Debug: sv_global shape = {sv_global.shape}")

                        plt.figure(figsize=(14, 9))
                        shap.summary_plot(
                            sv_global,
                            X_input_2d[:N_GLOBAL],
                            feature_names=feature_names,
                            show=False
                        )
                        plt.title(f"Global DeepExplainer - {model_name} - {dataset_name}")
                        plt.savefig(XAI_DIR / "global" / f"{dataset_name}_{model_name}_global.png",
                                    dpi=280, bbox_inches='tight')
                        plt.close()
                        global_done = True
                    except Exception as global_e:
                        print(f"   Global SHAP lỗi: {str(global_e)}")

            else:
                with open(MODEL_DIR / f"{dataset_name}_{model_name}.pkl", "rb") as f:
                    model = pickle.load(f)
                X_input_2d = X_test
                is_deep = False
                needs_3d = False

                if model_name in ["RandomForest", "LightGBM"]:
                    explainer = shap.TreeExplainer(model)
                    explainer_type = "TreeSHAP"
                else:
                    background = X_input_2d[:50]
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    explainer_type = "KernelSHAP"

                shap_values_raw = explainer.shap_values(X_input_2d[:N_GLOBAL])
                if isinstance(shap_values_raw, list):
                    sv_global = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
                else:
                    sv_global = shap_values_raw

                plt.figure(figsize=(14, 9))
                shap.summary_plot(sv_global, X_input_2d[:N_GLOBAL], feature_names=feature_names, show=False)
                plt.title(f"Global SHAP - {model_name} ({explainer_type}) - {dataset_name}")
                plt.savefig(XAI_DIR / "global" / f"{dataset_name}_{model_name}_global.png", dpi=280, bbox_inches='tight')
                plt.close()
                global_done = True

            # ─── LIME với wrapper reshape ────────────────────────────────────────
            print(f" → LIME Local...")
            try:
                lime_explainer = LimeTabularExplainer(
                    X_input_2d[:100],
                    feature_names=feature_names,
                    class_names=["Benign", "Attack"],
                    mode="classification"
                )

                # Wrapper predict để tự động reshape nếu model cần 3D
                def predict_wrapper(X):
                    if needs_3d:
                        # X từ LIME là 2D (n_samples, features) → reshape thành 3D
                        X_3d = X.reshape((X.shape[0], 1, X.shape[1]))
                        return model.predict(X_3d)
                    else:
                        return model.predict(X) if is_deep else model.predict_proba(X)

                for idx in [0, 10]:
                    data_instance = X_input_2d[idx]  # 1D vector
                    exp = lime_explainer.explain_instance(
                        data_instance,
                        predict_wrapper,  # dùng wrapper
                        num_features=15
                    )
                    exp.save_to_file(str(XAI_DIR / "local" / f"{dataset_name}_{model_name}_LIME_sample{idx}.html"))
                    with open(XAI_DIR / "local" / f"{dataset_name}_{model_name}_LIME_sample{idx}.txt", "w", encoding="utf-8") as f:
                        for feat, val in exp.as_list():
                            f.write(f"{feat}: {val:.4f}\n")
                lime_success = True
            except Exception as lime_e:
                print(f"   LIME lỗi: {str(lime_e)}")

            duration = round(time.time() - start_time, 1)
            comparison_table.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Explainer": explainer_type,
                "Global_SHAP": "✓" if global_done else "Skipped/Error",
                "Local_SHAP": "Skipped",
                "LIME": "✓" if lime_success else "Error",
                "Time(s)": duration
            })
            print(f" ✓ Hoàn thành trong {duration}s")

        except Exception as main_e:
            print(f" ✗ Lỗi tổng: {str(main_e)}")
            comparison_table.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Explainer": "Error",
                "Time(s)": 0
            })

# ====================== SAVE SUMMARY ======================
df = pd.DataFrame(comparison_table)
df.to_excel(XAI_DIR / "comparison" / "XAI_Comparison_Final2_lime_wrapper_fixed.xlsx", index=False)

print("\n" + "="*100)
print("HOÀN TẤT - LIME cho CNN/LSTM đã được fix bằng predictor wrapper")
print("="*100)
print(f"Kết quả lưu tại: {XAI_DIR}")