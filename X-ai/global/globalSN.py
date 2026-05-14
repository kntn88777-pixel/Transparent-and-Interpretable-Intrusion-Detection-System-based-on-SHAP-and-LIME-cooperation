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
SAVE_DIR = Path("XAI_Results3")

N_GLOBAL = 80
SAMPLE_INDICES = [0, 1, 5, 10, 15, 20, 30]


# ================= FIX FEATURE =================
def fix_feature_mismatch(X, expected_dim=30):
    if X.shape[1] < expected_dim:
        diff = expected_dim - X.shape[1]
        print(f" Padding thêm {diff} features")
        pad = np.zeros((X.shape[0], diff))
        X = np.hstack((X, pad))

    elif X.shape[1] > expected_dim:
        print("Cắt bớt feature")
        X = X[:, :expected_dim]

    return X


# ================= LOAD FEATURE NAMES =================
def load_feature_names():
    path1 = PREPROCESSED_DIR / "preprocessed_signal22_selected_features.pkl"
    path2 = PREPROCESSED_DIR / "preprocessed_signal22_feature_names.pkl"

    for path in [path1, path2]:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    names = pickle.load(f)
                print("Loaded feature names")
                return names  #  giữ full, không cắt 30
            except:
                pass

    print(" Using default feature names")
    return [f"Feature_{i}" for i in range(30)]


# ================= LOAD DATA =================
def load_data():
    X = np.load(PREPROCESSED_DIR / "preprocessed_signal22_X_test.npy")
    y = np.load(PREPROCESSED_DIR / "preprocessed_signal22_y_test.npy")

    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    #  FIX mismatch feature
    X = fix_feature_mismatch(X, 30)

    return X[:N_GLOBAL], y[:N_GLOBAL]


# ================= LOAD MODEL =================
def load_model(model_name):
    try:
        if model_name in ["DNN", "CNN", "LSTM"]:
            return tf.keras.models.load_model(MODEL_DIR / f"CIC17_{model_name}.h5")
        else:
            with open(MODEL_DIR / f"CIC17_{model_name}.pkl", "rb") as f:
                model = pickle.load(f)
                if hasattr(model, "n_jobs"):
                    model.n_jobs = 1  #  tránh crash
                return model
    except Exception as e:
        print(f"   ❌ Không load được model {model_name}: {e}")
        return None


# ================= MAIN =================
def run_local_explanation():
    X, y = load_data()
    feature_names = load_feature_names()

    print("=" * 100)
    print(" LOCAL EXPLANATION (FULL MODEL)")
    print("=" * 100)

    models = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

    for model_name in models:
        print(f"\n🔹 Model: {model_name}")
        model = load_model(model_name)
        if model is None:
            continue

        is_deep = model_name in ["DNN", "CNN", "LSTM"]
        needs_3d = model_name in ["CNN", "LSTM"]

        save_path = SAVE_DIR / "signal22" / model_name / "Local"
        os.makedirs(save_path, exist_ok=True)

        # ================= SHAP =================
        print("   → SHAP...")
        try:
            X_sample = X[:30]  #  giảm để tránh crash

            #  DÙNG KERNEL CHO TẤT CẢ → TRÁNH CRASH
            background = X_sample[:20]

            def predict_fn(x):
                if needs_3d:
                    x = x.reshape((x.shape[0], 1, x.shape[1]))
                return model.predict(x) if is_deep else (
                    model.predict_proba(x) if hasattr(model, "predict_proba") else model.predict(x)
                )

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1]

            for idx in SAMPLE_INDICES:
                if idx >= len(X_sample):
                    continue

                exp = shap.Explanation(
                    values=shap_values[idx],
                    base_values=expected_value,
                    data=X_sample[idx],
                    feature_names=feature_names
                )

                shap.plots.waterfall(exp, show=False)
                plt.savefig(save_path / f"SHAP_{idx}.png")
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
                if idx >= len(X):
                    continue

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

                exp.save_to_file(str(save_path / f"LIME_{idx}.html"))
                print(f"     ✓ LIME Sample {idx}")

        except Exception as e:
            print(f"     ❌ LIME lỗi: {e}")

    print("\nDONE FULL LOCAL XAI!")
    print(f"Saved at: {SAVE_DIR / 'signal22'}")


# ================= RUN =================
if __name__ == "__main__":
    print(" START LOCAL FULL MODEL...")
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')
    run_local_explanation()
