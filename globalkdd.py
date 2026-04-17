import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ================= PATH =================
PREPROCESSED_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
SAVE_DIR = Path("XAI_Results3")

N_GLOBAL = 80
SAMPLE_INDICES = [0, 1, 5, 10, 15, 20, 30]


# ================= LOAD FEATURE NAMES =================
def load_feature_names():
    path1 = PREPROCESSED_DIR / "preprocessed_nsl-kdd_selected_features.pkl"
    path2 = PREPROCESSED_DIR / "preprocessed_nsl-kdd_feature_names.pkl"

    if path1.exists():
        try:
            with open(path1, "rb") as f:
                names = pickle.load(f)
            print("✅ Loaded selected features")
            return names[:30]
        except:
            pass

    if path2.exists():
        try:
            with open(path2, "rb") as f:
                names = pickle.load(f)
            print("✅ Loaded feature names")
            return names[:30]
        except:
            pass

    print("⚠️ Using default feature names")
    return [f"Feature_{i}" for i in range(30)]


# ================= LOAD DATA =================
def load_data():
    X = np.load(PREPROCESSED_DIR / "preprocessed_nsl-kdd_X_test.npy")
    y = np.load(PREPROCESSED_DIR / "preprocessed_nsl-kdd_y_test.npy")

    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    return X[:N_GLOBAL], y[:N_GLOBAL]


# ================= LOAD MODEL =================
def load_model(model_name):
    try:
        if model_name in ["DNN", "CNN", "LSTM"]:
            return tf.keras.models.load_model(MODEL_DIR / f"nsl-kdd_{model_name}.h5")
        else:
            with open(MODEL_DIR / f"nsl-kdd_{model_name}.pkl", "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"❌ Load model lỗi {model_name}: {e}")
        return None


# ================= MAIN =================
def run_local_explanation():
    X, y = load_data()
    feature_names = load_feature_names()

    print("=" * 100)
    print("🔥 LOCAL EXPLANATION (FULL MODEL)")
    print("=" * 100)

    # 🔥 FULL MODELS
    models = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

    for model_name in models:
        print(f"\n🔹 Model: {model_name}")
        model = load_model(model_name)
        if model is None:
            continue

        is_deep = model_name in ["DNN", "CNN", "LSTM"]
        needs_3d = model_name in ["CNN", "LSTM"]

        save_path = SAVE_DIR / "nsl-kdd" / model_name / "Local"
        os.makedirs(save_path, exist_ok=True)

        # ================= SHAP =================
        print("   → SHAP...")
        try:
            # ===== chọn explainer đúng =====
            if model_name in ["RandomForest", "LightGBM"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

            elif model_name in ["DNN", "CNN"]:
                X_input = X.reshape((X.shape[0], 1, X.shape[1])) if needs_3d else X
                explainer = shap.DeepExplainer(model, X_input[:50])
                shap_values = explainer.shap_values(X_input)

            else:
                # AdaBoost, MLP → dùng KernelSHAP
                background = X[:30]
                predict_fn = (
                    lambda x: model.predict(x)
                    if is_deep else
                    (model.predict_proba if hasattr(model, "predict_proba") else model.predict)
                )
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X)

            # ===== xử lý output =====
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1]

            for idx in SAMPLE_INDICES:
                true_label = "Attack" if y[idx] == 1 else "Benign"

                data_input = X[idx]
                if needs_3d:
                    data_input = X[idx].reshape(1, 1, -1)

                exp = shap.Explanation(
                    values=shap_values[idx],
                    base_values=expected_value,
                    data=X[idx],
                    feature_names=feature_names
                )

                # ===== Waterfall =====
                shap.plots.waterfall(exp, max_display=15, show=False)
                plt.title(f"SHAP Waterfall - {model_name} - Sample {idx} ({true_label})")
                plt.savefig(save_path / f"SHAP_Waterfall_{idx}.png", dpi=200, bbox_inches='tight')
                plt.close()

                # ===== Force =====
                force = shap.force_plot(
                    expected_value,
                    shap_values[idx],
                    X[idx],
                    feature_names=feature_names
                )
                shap.save_html(str(save_path / f"SHAP_Force_{idx}.html"), force)

                # ===== Decision =====
                shap.decision_plot(
                    expected_value,
                    shap_values[idx],
                    X[idx],
                    feature_names=feature_names,
                    show=False
                )
                plt.title(f"SHAP Decision - {model_name} - Sample {idx}")
                plt.savefig(save_path / f"SHAP_Decision_{idx}.png", dpi=200, bbox_inches='tight')
                plt.close()

                print(f"     ✓ SHAP Sample {idx}")

        except Exception as e:
            print(f"     ❌ SHAP lỗi: {e}")

        # ================= LIME =================
        print("   → LIME...")
        try:
            from lime.lime_tabular import LimeTabularExplainer

            lime_explainer = LimeTabularExplainer(
                X,
                feature_names=feature_names,
                class_names=["Benign", "Attack"],
                mode="classification"
            )

            for idx in SAMPLE_INDICES:
                true_label = "Attack" if y[idx] == 1 else "Benign"

                def predict_fn(data):
                    if needs_3d:
                        data = data.reshape((data.shape[0], 1, data.shape[1]))
                    return model.predict(data) if is_deep else (
                        model.predict_proba(data) if hasattr(model, "predict_proba") else model.predict(data)
                    )

                exp = lime_explainer.explain_instance(
                    X[idx],
                    predict_fn,
                    num_features=15
                )

                html_file = save_path / f"LIME_{idx}.html"
                exp.save_to_file(str(html_file))

                print(f"     ✓ LIME Sample {idx}")

        except Exception as e:
            print(f"     ❌ LIME lỗi: {e}")

    print("\n🎯 DONE FULL LOCAL XAI!")
    print(f"📂 Saved at: {SAVE_DIR / 'nsl-kdd'}")


# ================= RUN =================
if __name__ == "__main__":
    print("🔥 START LOCAL FULL MODEL...")
    tf.get_logger().setLevel('ERROR')
    run_local_explanation()