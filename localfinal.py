import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

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
SAVE_DIR = Path("XAI_LOCAL1")

SAMPLE_INDICES = [0, 1, 5, 10, 15, 20, 30]


# ================= FEATURE =================
def load_feature_names():
    path = PREPROCESSED_DIR / "preprocessed_cic17_feature_names.pkl"

    if path.exists():
        with open(path, "rb") as f:
            names = pickle.load(f)
        print(f"✅ Loaded feature names: {len(names)}")
        return names

    return [f"Feature_{i}" for i in range(41)]


# ================= DATA =================
def load_data():
    X = np.load(PREPROCESSED_DIR / "preprocessed_cic17_X_test.npy")
    y = np.load(PREPROCESSED_DIR / "preprocessed_cic17_y_test.npy")

    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    print(f"✅ Data loaded: {X.shape}")
    return X, y


# ================= MODEL =================
def load_model(model_name):
    try:
        if model_name in ["DNN", "CNN", "LSTM"]:
            return tf.keras.models.load_model(MODEL_DIR / f"cic17_{model_name}.h5")
        else:
            with open(MODEL_DIR / f"cic17_{model_name}.pkl", "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"❌ Load lỗi {model_name}: {e}")
        return None


# ================= PREDICT FUNCTION =================
def get_predict_fn(model, is_deep, needs_3d):
    def predict_fn(data):
        if needs_3d:
            data = data.reshape((data.shape[0], 1, data.shape[1]))

        if is_deep:
            return model.predict(data, verbose=0)
        else:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(data)
            else:
                preds = model.predict(data)
                return np.vstack([1 - preds, preds]).T

    return predict_fn


# ================= SHAP LOCAL =================
def run_shap(model, model_name, X, y, feature_names, save_path, is_deep, needs_3d):

    print("   → SHAP Local...")

    try:
        # ===== CHỈ LẤY SAMPLE =====
        X_sample = X[SAMPLE_INDICES]

        # ===== EXPLAINER =====
        if model_name in ["RandomForest", "LightGBM"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

        elif is_deep:
            X_input = X.reshape((X.shape[0], 1, X.shape[1])) if needs_3d else X
            X_sample_input = X_input[SAMPLE_INDICES]

            explainer = shap.DeepExplainer(model, X_input[:50])
            shap_values = explainer.shap_values(X_sample_input)

        else:
            background = X[np.random.choice(X.shape[0], 30, replace=False)]
            predict_fn = get_predict_fn(model, is_deep, needs_3d)

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample)

        # ===== FIX BINARY =====
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        # ===== WATERFALL =====
        for i, idx in enumerate(SAMPLE_INDICES):
            label = "Attack" if y[idx] == 1 else "Normal"

            exp = shap.Explanation(
                values=shap_values[i],
                base_values=expected_value,
                data=X[idx],
                feature_names=feature_names
            )

            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(exp, max_display=15, show=False)
            plt.title(f"{model_name} - SHAP Sample {idx} ({label})")

            plt.savefig(save_path / f"SHAP_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"     ✓ SHAP {idx}")

    except Exception as e:
        print(f"     ❌ SHAP lỗi: {e}")


# ================= LIME LOCAL =================
def run_lime(model, model_name, X, y, feature_names, save_path, is_deep, needs_3d):

    print("   → LIME Local...")

    try:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            X,
            feature_names=feature_names,
            class_names=["Normal", "Attack"],
            mode="classification"
        )

        predict_fn = get_predict_fn(model, is_deep, needs_3d)

        for idx in SAMPLE_INDICES:
            exp = explainer.explain_instance(
                X[idx],
                predict_fn,
                num_features=15
            )

            # HTML
            exp.save_to_file(str(save_path / f"LIME_{idx}.html"))

            # PNG
            fig = exp.as_pyplot_figure()
            plt.title(f"{model_name} - LIME Sample {idx}")
            plt.tight_layout()

            plt.savefig(save_path / f"LIME_{idx}.png", dpi=300)
            plt.close()

            print(f"     ✓ LIME {idx}")

    except Exception as e:
        print(f"     ❌ LIME lỗi: {e}")


# ================= MAIN =================
def run_local_xai():

    X, y = load_data()
    feature_names = load_feature_names()

    print("=" * 100)
    print("🔥 LOCAL XAI (SHAP + LIME) - FIXED")
    print("=" * 100)

    models = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

    for model_name in models:

        print(f"\n🔹 Model: {model_name}")

        model = load_model(model_name)
        if model is None:
            continue

        is_deep = model_name in ["DNN", "CNN", "LSTM"]
        needs_3d = model_name in ["CNN", "LSTM"]

        save_path = SAVE_DIR / "cic17" / model_name
        os.makedirs(save_path, exist_ok=True)

        run_shap(model, model_name, X, y, feature_names, save_path, is_deep, needs_3d)
        run_lime(model, model_name, X, y, feature_names, save_path, is_deep, needs_3d)

    print("\n🎯 DONE LOCAL XAI!")


# ================= RUN =================
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    run_local_xai()