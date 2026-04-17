import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
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
SAVE_DIR = Path("XAI_Results_4")

N_GLOBAL = 80

# ================= TÊN FEATURE GỐC CIC17 =================
ALL_COLUMNS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min", "Label"
]

EXCLUDE_COLS = {"Flow ID", "Source IP", "Source Port", "Destination IP", "Protocol", "Timestamp", "Label"}

FEATURE_NAMES_CIC17 = [col for col in ALL_COLUMNS if col not in EXCLUDE_COLS]

print(f"→ Số feature sau khi lọc: {len(FEATURE_NAMES_CIC17)}")


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
        if model_name in ["LSTM"]:
            model_path = MODEL_DIR / f"CIC17_{model_name}.h5"
            return tf.keras.models.load_model(model_path, compile=False)
        else:
            model_path = MODEL_DIR / f"CIC17_{model_name}.pkl"
            with open(model_path, "rb") as f:
                return pickle.load(f)

    except Exception as e:
        print(f"❌ Không load được model {model_name}: {e}")
        return None


# ================= DEEPSHAP FALLBACK FOR LSTM =================
def compute_lstm_deepshap_proxy(model, X):
    background = X[:20]

    def predict_fn(data):
        data_3d = data.reshape((data.shape[0], 1, data.shape[1]))
        pred = model.predict(data_3d, verbose=0)

        if pred.ndim == 1:
            pred = np.vstack([1 - pred, pred]).T
        elif pred.shape[1] == 1:
            pred = np.hstack([1 - pred, pred])

        return pred

    explainer = shap.KernelExplainer(
        predict_fn,
        background
    )

    sv = explainer.shap_values(
        X,
        nsamples=100
    )

    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]

    return sv


# ================= MAIN =================
def run_all_models_cic17():
    dataset_name = "CIC17"

    X_test, y_test = load_dataset()
    X = X_test[:N_GLOBAL]

    print("=" * 100)
    print(f"ĐANG CHẠY XAI CHO CIC17 - {N_GLOBAL} samples")
    print(f"Số feature: {X.shape[1]} | Tên feature gốc: {len(FEATURE_NAMES_CIC17)}")
    print("=" * 100)

    MODELS = ["LSTM"]
    EXPLAINERS = ["TreeSHAP", "KernelSHAP", "DeepSHAP", "LIME"]

    for model_name in MODELS:
        print(f"\n→ MODEL: {model_name}")

        model = load_model(model_name)
        if model is None:
            continue

        is_deep = model_name in ["LSTM"]
        needs_3d = model_name in ["LSTM"]

        X_input = X.reshape((X.shape[0], 1, X.shape[1])) if needs_3d else X

        for explainer_type in EXPLAINERS:

            if explainer_type == "DeepSHAP" and model_name not in ["DNN", "CNN", "LSTM"]:
                continue

            if explainer_type == "KernelSHAP" and model_name in ["CNN", "LSTM"]:
                continue

            print(f"   → Explainer: {explainer_type}")

            try:
                # ================= SHAP =================
                if explainer_type == "TreeSHAP":
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(X)

                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]

                elif explainer_type == "KernelSHAP":
                    background = X[:30]

                    explainer = shap.KernelExplainer(
                        (lambda x: model.predict(x)) if is_deep else (lambda x: model.predict_proba(x)),
                        background
                    )

                    sv = explainer.shap_values(X)

                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]

                elif explainer_type == "DeepSHAP":

                    if model_name == "LSTM":
                        sv = compute_lstm_deepshap_proxy(model, X)

                    else:
                        explainer = shap.DeepExplainer(model, X_input[:50])
                        sv = explainer.shap_values(X_input)

                        if isinstance(sv, list):
                            sv = sv[1] if len(sv) > 1 else sv[0]

                        if sv.ndim == 3:
                            sv = sv[:, 0, :]

                elif explainer_type == "LIME":
                    from lime.lime_tabular import LimeTabularExplainer

                    lime_exp = LimeTabularExplainer(
                        X,
                        feature_names=FEATURE_NAMES_CIC17,
                        class_names=["Benign", "Attack"],
                        mode="classification"
                    )

                    lime_values = np.zeros((len(X), X.shape[1]))

                    def predict_fn(data):
                        if needs_3d:
                            data = data.reshape((data.shape[0], 1, data.shape[1]))

                        return model.predict(data) if is_deep else model.predict_proba(data)

                    for i in range(len(X)):
                        exp = lime_exp.explain_instance(
                            X[i],
                            predict_fn,
                            num_features=X.shape[1]
                        )

                        for feat, val in exp.as_list():
                            idx = int(re.findall(r'\d+', feat)[0])
                            lime_values[i, idx] = val

                    sv = lime_values

                # ================= SAVE =================
                save_path = SAVE_DIR / dataset_name / model_name
                os.makedirs(save_path, exist_ok=True)

                plt.figure(figsize=(11, 7.5))

                shap.summary_plot(
                    sv,
                    X,
                    feature_names=FEATURE_NAMES_CIC17,
                    show=False
                )

                plt.title(f"{explainer_type} - {model_name} - CIC17", fontsize=13)
                plt.tight_layout()

                filename = f"{explainer_type}_CIC17.png"

                plt.savefig(
                    save_path / filename,
                    dpi=160,
                    bbox_inches='tight'
                )

                plt.close()

                print(f"   ✓ Đã lưu: {filename}")

            except Exception as e:
                print(f"   ❌ Lỗi {explainer_type}: {e}")

    print("\n🎯 HOÀN THÀNH TẤT CẢ MODEL CHO CIC17!")


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    run_all_models_cic17()