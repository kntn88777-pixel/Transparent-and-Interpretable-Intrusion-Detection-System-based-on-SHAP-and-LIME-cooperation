import matplotlib

matplotlib.use('Agg')  # BẮT BUỘC phải đặt trước khi import pyplot

import time
import numpy as np
import pandas as pd
import os
import pickle
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Tắt các warning gây phiền
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ========================= CONFIG =========================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
PLOT_DIR = Path("plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR / "confusion", exist_ok=True)
os.makedirs(PLOT_DIR / "roc", exist_ok=True)
os.makedirs(PLOT_DIR / "loss", exist_ok=True)

DATASET = {
    "name": "SIGNAL2022",
    "X_train": "signal22_X_train.npy",
    "X_test": "signal22_X_test.npy",
    "Y_train": "signal22_y_train.npy",
    "Y_test": "signal22_y_test.npy"
}


# ========================= HELPER FUNCTIONS =========================
def compute_ids_metrics(y_true, y_pred):
    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        FAR = FP / (FP + TN) if (FP + TN) != 0 else 0
        FAP = FP / (FP + TP) if (FP + TP) != 0 else 0
        FAF1 = (2 * TP) / (2 * TP + FN + 2 * FP) if (2 * TP + FN + 2 * FP) != 0 else 0
        DR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    else:
        FAR = FAP = FAF1 = DR = FPR = 0
    return FAR, FAP, FAF1, DR, FPR


def load_dataset():
    print(f"\n→ Loading dataset: {DATASET['name']}")
    X_train = np.load(DATA_DIR / DATASET["X_train"])
    X_test = np.load(DATA_DIR / DATASET["X_test"])
    y_train = np.load(DATA_DIR / DATASET["Y_train"])
    y_test = np.load(DATA_DIR / DATASET["Y_test"])

    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    print(f" Shape → X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f" Classes: {np.unique(y_train)}")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"SIGNAL2022 – {model_name} Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion" / f"SIGNAL2022_{model_name}_cm.png", dpi=120)
    plt.close('all')


def plot_roc(y_true, y_score, model_name):
    y_true_bin = (y_true != 0).astype(int)
    if y_score.ndim == 2:
        if y_score.shape[1] == 2:
            y_score_bin = y_score[:, 1]
        else:
            y_score_bin = y_score[:, 1:].max(axis=1)
    else:
        y_score_bin = y_score.ravel()

    y_score_bin = np.clip(y_score_bin, 0, 1)
    fpr, tpr, _ = roc_curve(y_true_bin, y_score_bin)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"SIGNAL2022 – {model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc" / f"SIGNAL2022_{model_name}_roc.png", dpi=150)
    plt.close('all')


def plot_loss(history, model_name):
    plt.figure(figsize=(6, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"SIGNAL2022 - {model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "loss" / f"SIGNAL2022_{model_name}_loss.png", dpi=120)
    plt.close('all')


# ========================= DEEP LEARNING MODELS =========================
def train_dnn(X_train, y_train, X_test):
    num_classes = len(np.unique(y_train))
    model = Sequential([
        Dense(256, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    pred = model.predict(X_test, verbose=0)
    return model, np.argmax(pred, axis=1), pred, history


def train_cnn(X_train, y_train, X_test):
    num_classes = len(np.unique(y_train))
    X_train_r = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_r = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_train_cat = to_categorical(y_train)

    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=(1, X_train.shape[1])),
        Conv1D(128, 3, activation='relu', padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        X_train_r, y_train_cat,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    pred = model.predict(X_test_r, verbose=0)
    return model, np.argmax(pred, axis=1), pred, history


def train_lstm(X_train, y_train, X_test):
    num_classes = len(np.unique(y_train))
    X_train_r = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_r = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_train_cat = to_categorical(y_train)

    model = Sequential([
        LSTM(96, input_shape=(1, X_train.shape[1])),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        X_train_r, y_train_cat,
        epochs=15,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    pred = model.predict(X_test_r, verbose=0)
    return model, np.argmax(pred, axis=1), pred, history


# ========================= TRADITIONAL MODELS =========================
def get_models(num_classes):
    is_multiclass = num_classes > 2
    lgb_params = {
        'n_estimators': 150,
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_jobs': 4,
        'verbose': -1,
        'random_state': 42,
    }
    if is_multiclass:
        lgb_params['objective'] = 'multiclass'
        lgb_params['num_class'] = num_classes
    else:
        lgb_params['objective'] = 'binary'

    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=4, random_state=42),
        "LightGBM": lgb.LGBMClassifier(**lgb_params),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, early_stopping=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=80, random_state=42),
        "DNN": "DNN",
        "CNN": "CNN",
        "LSTM": "LSTM",
    }


# ========================= MAIN =========================
def main():
    X_train, X_test, y_train, y_test = load_dataset()
    num_classes = len(np.unique(y_train))
    models = get_models(num_classes)

    results = []

    for name, model in models.items():
        print(f"\n→ Training: {name} on SIGNAL2022")
        start = time.time()

        try:
            if name in ["DNN", "CNN", "LSTM"]:
                if name == "DNN":
                    model_obj, y_pred, y_score, history = train_dnn(X_train, y_train, X_test)
                elif name == "CNN":
                    model_obj, y_pred, y_score, history = train_cnn(X_train, y_train, X_test)
                else:
                    model_obj, y_pred, y_score, history = train_lstm(X_train, y_train, X_test)

                # Lưu model Keras (dùng format mới)
                model_obj.save(MODEL_DIR / f"SIGNAL2022_{name}.keras")
                plot_loss(history, name)

            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                with open(MODEL_DIR / f"SIGNAL2022_{name}.pkl", "wb") as f:
                    pickle.dump(model, f)

            train_time = time.time() - start
            print(f" → Done in {train_time:.1f} seconds")

            # Tính metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            FAR, FAP, FAF1, DR, FPR = compute_ids_metrics(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, name)
            if y_score is not None:
                plot_roc(y_test, y_score, name)

            results.append({
                "Dataset": "SIGNAL2022",
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "DetectionRate": DR,
                "FPR": FPR,
                "FAF1": FAF1,
                "FAR": FAR,
                "FAP": FAP,
                "TrainTime": train_time
            })

        except Exception as e:
            print(f"❌ Lỗi khi train {name}: {str(e)}")
            continue

    # Lưu kết quả
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_excel("results/SIGNAL2022_results.xlsx", index=False)

    print("\n" + "=" * 70)
    print("HOÀN TẤT TRAINING SIGNAL2022!")
    print("Kết quả đã lưu tại: results/SIGNAL2022_results.xlsx")
    print("Model lưu tại: saved_models/")
    print("Biểu đồ lưu tại: plots/")
    print("=" * 70)


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()