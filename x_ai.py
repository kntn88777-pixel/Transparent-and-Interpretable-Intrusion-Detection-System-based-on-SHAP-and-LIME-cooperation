import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import re
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ================= PATH =================
PREPROCESSED_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
SAVE_DIR = Path("XAI_Results3")

N_GLOBAL = 80


# ================= LOAD 30 FEATURE NAMES TỪ FILE .PKL =================
def load_selected_feature_names():
    """Ưu tiên load từ selected_features.pkl, fallback sang feature_names.pkl"""
    # Thử file selected_features trước
    selected_path = PREPROCESSED_DIR / "preprocessed_cic17_selected_features.pkl"
    if selected_path.exists():
        try:
            with open(selected_path, "rb") as f:
                feature_names = pickle.load(f)
            if isinstance(feature_names, list):
                print(f"✅ Đã load 30 feature names từ: preprocessed_cic17_selected_features.pkl")
                print(f"   Số lượng: {len(feature_names)}")
                return feature_names
        except:
            pass

    # Thử file feature_names.pkl
    names_path = PREPROCESSED_DIR / "preprocessed_cic17_feature_names.pkl"
    if names_path.exists():
        try:
            with open(names_path, "rb") as f:
                feature_names = pickle.load(f)
            if isinstance(feature_names, list):
                print(f"✅ Đã load feature names từ: preprocessed_cic17_feature_names.pkl")
                # Nếu nhiều hơn 30, có thể cần cắt theo selector, nhưng tạm lấy hết
                return feature_names
        except:
            pass

    print("⚠️ Không load được file .pkl nào chứa tên feature → Dùng tên mặc định")
    return None


# ================= LOAD DATA =================
def load_dataset():
    X = np.load(PREPROCESSED_DIR / "preprocessed_cic17_X_test.npy")
    y = np.load(PREPROCESSED_DIR / "preprocessed_cic17_y_test.npy")
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    return X, y


# ================= LOAD MODEL =================
def load_model(model_name):
    try:
        if model_name in ["DNN", "CNN", "LSTM"]:
            return tf.keras.models.load_model(MODEL_DIR / f"CIC17_{model_name}.h5")
        else:
            with open(MODEL_DIR / f"CIC17_{model_name}.pkl", "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"   ❌ Không load được model {model_name}: {e}")
        return None


# ================= MAIN =================
def run_all_models_cic17():
    dataset_name = "CIC17"
    X_test, y_test = load_dataset()
    X = X_test[:N_GLOBAL]

    # Load tên 30 feature từ file .pkl
    feature_names = load_selected_feature_names()

    if feature_names is None or len(feature_names) == 0:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # Đảm bảo số lượng khớp với X
    feature_names = feature_names[:X.shape[1]]

    print("=" * 100)
    print(f"ĐANG CHẠY XAI CHO CIC17 - {N_GLOBAL} samples")
    print(f"Số feature trong X_test.npy : {X.shape[1]}")
    print(f"Số tên feature đang dùng   : {len(feature_names)}")
    print(f"10 feature đầu: {feature_names[:10]}")
    print("=" * 100)

    MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]
    EXPLAINERS = ["TreeSHAP", "KernelSHAP", "DeepSHAP", "LIME"]

    for model_name in MODELS:
        print(f"\n→ MODEL: {model_name}")

        model = load_model(model_name)
        if model is None:
            continue

        is_deep = model_name in ["DNN", "CNN", "LSTM"]
        needs_3d = model_name in ["CNN", "LSTM"]
        X_input = X.reshape((X.shape[0], 1, X.shape[1])) if needs_3d else X

        for explainer_type in EXPLAINERS:
            if explainer_type == "TreeSHAP" and model_name not in ["RandomForest", "LightGBM"]: continue
            if explainer_type == "DeepSHAP" and model_name not in ["DNN", "CNN"]: continue
            if explainer_type == "KernelSHAP" and model_name in ["CNN", "LSTM"]: continue

            print(f"   → Explainer: {explainer_type}")

            try:
                if explainer_type == "TreeSHAP":
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(X)
                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]

                elif explainer_type == "KernelSHAP":
                    explainer = shap.KernelExplainer(
                        (lambda x: model.predict(x)) if is_deep else (lambda x: model.predict_proba(x)),
                        X[:30]
                    )
                    sv = explainer.shap_values(X)
                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]

                elif explainer_type == "DeepSHAP":
                    explainer = shap.DeepExplainer(model, X_input[:50])
                    sv = explainer.shap_values(X_input)
                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]
                    if sv.ndim == 3:
                        sv = sv[:, 0, :]

                elif explainer_type == "LIME":
                    from lime.lime_tabular import LimeTabularExplainer
                    lime_exp = LimeTabularExplainer(
                        X, feature_names=feature_names,
                        class_names=["Benign", "Attack"], mode="classification"
                    )
                    lime_values = np.zeros((len(X), X.shape[1]))

                    def predict_fn(data):
                        if needs_3d:
                            data = data.reshape((data.shape[0], 1, data.shape[1]))
                        return model.predict(data) if is_deep else model.predict_proba(data)

                    for i in range(len(X)):
                        exp = lime_exp.explain_instance(X[i], predict_fn, num_features=X.shape[1])
                        for feat, val in exp.as_list():
                            match = re.search(r'(\d+)', feat)
                            if match:
                                idx = int(match.group(1))
                                if idx < X.shape[1]:
                                    lime_values[i, idx] = val
                    sv = lime_values

                # Save
                save_path = SAVE_DIR / dataset_name / model_name
                os.makedirs(save_path, exist_ok=True)

                plt.figure(figsize=(12, 8))
                shap.summary_plot(sv, X, feature_names=feature_names, show=False)
                plt.title(f"{explainer_type} - {model_name} - CIC17", fontsize=13)
                plt.tight_layout()

                filename = f"{explainer_type}_CIC17.png"
                plt.savefig(save_path / filename, dpi=160, bbox_inches='tight')
                plt.close()

                print(f"   ✓ Đã lưu: {filename}")

            except Exception as e:
                print(f"   ❌ Lỗi {explainer_type}: {e}")

    print("\n🎯 HOÀN THÀNH TẤT CẢ MODEL CHO CIC17!")


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    run_all_models_cic17()