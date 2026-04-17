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


# ================= FIX FEATURE =================
def fix_feature_mismatch(X, expected_dim=30):
    if X.shape[1] < expected_dim:
        diff = expected_dim - X.shape[1]
        print(f"⚠️ Padding thêm {diff} features")
        pad = np.zeros((X.shape[0], diff))
        X = np.hstack((X, pad))

    elif X.shape[1] > expected_dim:
        print("⚠️ Cắt bớt feature")
        X = X[:, :expected_dim]

    return X


# ================= FEATURE =================
def load_feature_names():
    path1 = PREPROCESSED_DIR / "preprocessed_signal22_selected_features.pkl"
    path2 = PREPROCESSED_DIR / "preprocessed_signal22_feature_names.pkl"

    for path in [path1, path2]:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    names = pickle.load(f)
                print("✅ Loaded feature names")
                return names  # 🔥 giữ full, không cắt 30
            except:
                pass

    print("⚠️ Using default feature names")
    return [f"Feature_{i}" for i in range(30)]

# ================= DATA =================
def load_data():
    X = np.load(PREPROCESSED_DIR / "preprocessed_signal22_X_test.npy")
    y = np.load(PREPROCESSED_DIR / "preprocessed_signal22_y_test.npy")

    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    # 🔥 FIX mismatch feature
    X = fix_feature_mismatch(X, 30)

    return X[:N_GLOBAL], y[:N_GLOBAL]

# ================= MODEL =================
def load_model(model_name):
    try:
        if model_name in ["DNN", "CNN", "LSTM"]:
            return tf.keras.models.load_model(MODEL_DIR / f"CIC17_{model_name}.h5")
        else:
            with open(MODEL_DIR / f"CIC17_{model_name}.pkl", "rb") as f:
                model = pickle.load(f)
                if hasattr(model, "n_jobs"):
                    model.n_jobs = 1  # 🔥 tránh crash
                return model
    except Exception as e:
        print(f"   ❌ Không load được model {model_name}: {e}")
        return None


# ================= MAIN =================
def run_local_explanation():
    X, y = load_data()
    feature_names = load_feature_names()

    print("=" * 100)
    print("🔥 FULL XAI (SHAP + LIME HTML + PNG)")
    print("=" * 100)

    models = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

    for model_name in models:
        print(f"\n🔹 Model: {model_name}")
        model = load_model(model_name)
        if model is None:
            continue

        is_deep = model_name in ["DNN", "CNN", "LSTM"]
        needs_3d = model_name in ["CNN", "LSTM"]

        save_path = SAVE_DIR / "cic17" / model_name / "Local"
        os.makedirs(save_path, exist_ok=True)

        # ================= SHAP =================
        print("   → SHAP...")
        try:
            if model_name in ["RandomForest", "LightGBM"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

            elif is_deep:
                X_input = X.reshape((X.shape[0], 1, X.shape[1])) if needs_3d else X
                explainer = shap.DeepExplainer(model, X_input[:50])
                shap_values = explainer.shap_values(X_input)

            else:
                background = X[:30]

                def predict_fn(x):
                    if needs_3d:
                        x = x.reshape((x.shape[0], 1, x.shape[1]))
                    return model.predict(x) if is_deep else (
                        model.predict_proba(x) if hasattr(model, "predict_proba") else model.predict(x)
                    )

                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X)
            # ===== GLOBAL SHAP =====
            try:
                print("   → SHAP Global...")

                # Summary (beeswarm)
                plt.figure()
                shap.summary_plot(
                    shap_values,
                    X,
                    feature_names=feature_names,
                    show=False
                )
                plt.title(f"{model_name} - SHAP Summary")
                plt.savefig(save_path / "SHAP_Summary.png", dpi=300, bbox_inches='tight')
                plt.close()

                # Bar global
                plt.figure()
                shap.summary_plot(
                    shap_values,
                    X,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False
                )
                plt.title(f"{model_name} - SHAP Global Importance")
                plt.savefig(save_path / "SHAP_Bar_Global.png", dpi=300, bbox_inches='tight')
                plt.close()

                print("     ✓ SHAP Global Done")

            except Exception as e:
                print(f"     ❌ SHAP Global lỗi: {e}")
            # ===== FIX CLASS =====
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1]

            # ===== PLOT =====
            for idx in SAMPLE_INDICES:
                true_label = "Attack" if y[idx] == 1 else "Normal"

                exp = shap.Explanation(
                    values=shap_values[idx],
                    base_values=expected_value,
                    data=X[idx],
                    feature_names=feature_names
                )

                plt.figure(figsize=(8, 6))
                shap.plots.waterfall(exp, max_display=15, show=False)
                plt.title(f"{model_name} - Sample {idx} ({true_label})")
                plt.savefig(save_path / f"SHAP_{idx}.png", dpi=300, bbox_inches='tight')
                plt.close()

                print(f"     ✓ SHAP {idx}")

        except Exception as e:
            print(f"     ❌ SHAP lỗi: {e}")

        # ================= LIME =================
        # ================= LIME =================
        print("   → LIME (HTML + PNG)...")
        try:
            from lime.lime_tabular import LimeTabularExplainer

            lime_explainer = LimeTabularExplainer(
                X,
                feature_names=feature_names,
                class_names=["Normal", "Attack"],
                mode="classification"
            )

            for idx in SAMPLE_INDICES:

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

                # ===== HTML (interactive) =====
                html_file = save_path / f"LIME_{idx}.html"
                exp.save_to_file(str(html_file))

                # ===== PNG (KHÔNG CẦN selenium) =====
                fig = exp.as_pyplot_figure()
                plt.title(f"{model_name} - LIME Sample {idx}")
                plt.tight_layout()
                plt.savefig(save_path / f"LIME_{idx}.png", dpi=300)
                plt.close()

                print(f"     ✓ LIME {idx}")

        except Exception as e:
            print(f"     ❌ LIME lỗi: {e}")
    print("\n🎯 DONE FULL XAI!")

# ================= RUN =================
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    run_local_explanation()