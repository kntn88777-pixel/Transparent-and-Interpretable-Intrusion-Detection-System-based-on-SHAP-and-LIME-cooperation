import numpy as np
import shap
import pickle
import tensorflow as tf
from pathlib import Path

# ================= PATH =================
PREPROCESSED_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path(r"D:\luanvsn\.venv\final\saved_models")

models = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

# ================= LOAD DATA =================
def load_data():
    X = np.load(PREPROCESSED_DIR / "preprocessed_cic17_X_test.npy")
    y = np.load(PREPROCESSED_DIR / "preprocessed_cic17_y_test.npy")
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    return X[:50], y[:50]  # giảm để tránh lag


# ================= LOAD MODEL =================
def load_model(model_name):
    if model_name in ["DNN", "CNN", "LSTM"]:
        return tf.keras.models.load_model(MODEL_DIR / f"CIC17_{model_name}.h5")
    else:
        with open(MODEL_DIR / f"CIC17_{model_name}.pkl", "rb") as f:
            model = pickle.load(f)
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
            return model


# ================= METRICS =================
def sparsity(explanations):
    return np.mean(np.sum(explanations == 0, axis=1) / explanations.shape[1])

def complexity(explanations):
    return np.mean(np.count_nonzero(explanations, axis=1))

def stability(explanations):
    diffs = []
    for i in range(len(explanations)-1):
        diffs.append(np.linalg.norm(explanations[i] - explanations[i+1]))
    return np.mean(diffs)


# ================= MAIN =================
X, y = load_data()

for model_name in models:
    print(f"\n🔹 {model_name}")
    model = load_model(model_name)

    is_deep = model_name in ["DNN", "CNN", "LSTM"]
    needs_3d = model_name in ["CNN", "LSTM"]

    # ================= SHAP =================
    try:
        if model_name in ["RandomForest", "LightGBM"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

        else:
            background = X[:20]

            def predict_fn(x):
                if needs_3d:
                    x = x.reshape((x.shape[0], 1, x.shape[1]))
                return model.predict(x) if is_deep else (
                    model.predict_proba(x) if hasattr(model, "predict_proba") else model.predict(x)
                )

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_local = shap_values

        print("   SHAP:",
              "Sparsity =", sparsity(shap_local),
              "| Complexity =", complexity(shap_local),
              "| Stability =", stability(shap_local))

    except Exception as e:
        print("   ❌ SHAP lỗi:", e)


    # ================= LIME =================
    try:
        from lime.lime_tabular import LimeTabularExplainer

        lime_explainer = LimeTabularExplainer(
            X,
            mode="classification"
        )

        lime_local = []

        for i in range(len(X)):

            def predict_fn(data):
                if needs_3d:
                    data = data.reshape((data.shape[0], 1, data.shape[1]))
                return model.predict(data) if is_deep else (
                    model.predict_proba(data) if hasattr(model, "predict_proba") else model.predict(data)
                )

            exp = lime_explainer.explain_instance(
                X[i],
                predict_fn,
                num_features=X.shape[1]
            )

            weights = dict(exp.as_list())
            vector = np.zeros(X.shape[1])

            for k, v in weights.items():
                try:
                    idx = int(k.split(' ')[0].replace('Feature_', '').replace('feature_', ''))
                    vector[idx] = v
                except:
                    pass

            lime_local.append(vector)

        lime_local = np.array(lime_local)

        print("   LIME:",
              "Sparsity =", sparsity(lime_local),
              "| Complexity =", complexity(lime_local),
              "| Stability =", stability(lime_local))

    except Exception as e:
        print("   ❌ LIME lỗi:", e)