import time
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from tensorflow.keras.utils import to_categorical


# ────────────────────────────────────────────────
# DATA PATH
# ────────────────────────────────────────────────

DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")

MODEL_DIR = Path("saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


DATASET_FILES = [
    {"name": "CIC17",
     "X_train": "preprocessed_cic17_X_train.npy",
     "X_test": "preprocessed_cic17_X_test.npy",
     "Y_train": "preprocessed_cic17_y_train.npy",
     "Y_test": "preprocessed_cic17_y_test.npy"},

    {"name": "NSL-KDD",
     "X_train": "preprocessed_nsl-kdd_X_train.npy",
     "X_test": "preprocessed_nsl-kdd_X_test.npy",
     "Y_train": "preprocessed_nsl-kdd_y_train.npy",
     "Y_test": "preprocessed_nsl-kdd_y_test.npy"},

    {"name": "SIGNAL2022",
     "X_train": "signal22_X_train.npy",
     "X_test": "signal22_X_test.npy",
     "Y_train": "signal22_y_train.npy",
     "Y_test": "signal22_y_test.npy"},
]


# ────────────────────────────────────────────────
# IDS METRICS
# ────────────────────────────────────────────────

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


# ────────────────────────────────────────────────
# LOAD DATASET
# ────────────────────────────────────────────────

def load_dataset(ds):

    print("\nLoading:", ds["name"])

    X_train = np.load(DATA_DIR / ds["X_train"])
    X_test = np.load(DATA_DIR / ds["X_test"])

    y_train = np.load(DATA_DIR / ds["Y_train"])
    y_test = np.load(DATA_DIR / ds["Y_test"])

    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

    return X_train, X_test, y_train, y_test


# ────────────────────────────────────────────────
# PLOT FUNCTIONS
# ────────────────────────────────────────────────

def plot_confusion_matrix(cm, dataset, model):

    os.makedirs("plots/confusion", exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{dataset} - {model} Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    plt.savefig(f"plots/confusion/{dataset}_{model}_cm.png")
    plt.close()


def plot_roc(y_true, y_score, dataset, model):

    os.makedirs("plots/roc", exist_ok=True)

    y_true_bin = (y_true != 0).astype(int)

    if y_score.ndim == 2 and y_score.shape[1] > 1:
        y_score_bin = y_score[:, 1:].max(axis=1)
    else:
        y_score_bin = y_score.ravel()

    y_score_bin = np.clip(y_score_bin, 0, 1)

    fpr, tpr, _ = roc_curve(y_true_bin, y_score_bin)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")

    plt.title(f"{dataset} - {model} ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(f"plots/roc/{dataset}_{model}_roc.png", dpi=150)
    plt.close()


# ────────────────────────────────────────────────
# DEEP LEARNING MODELS
# ────────────────────────────────────────────────

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

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=0)

    pred = model.predict(X_test, verbose=0)

    return model, np.argmax(pred, axis=1), pred


def train_cnn(X_train, y_train, X_test):

    num_classes = len(np.unique(y_train))

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    y_train = to_categorical(y_train)

    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same',
               input_shape=(1, X_train.shape[2])),

        Conv1D(128, 3, activation='relu', padding='same'),

        Flatten(),

        Dense(128, activation='relu'),

        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=15, batch_size=128, verbose=0)

    pred = model.predict(X_test, verbose=0)

    return model, np.argmax(pred, axis=1), pred


def train_lstm(X_train, y_train, X_test):

    num_classes = len(np.unique(y_train))

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    y_train = to_categorical(y_train)

    model = Sequential([
        LSTM(128, input_shape=(1, X_train.shape[2])),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=12, batch_size=128, verbose=0)

    pred = model.predict(X_test, verbose=0)

    return model, np.argmax(pred, axis=1), pred


# ────────────────────────────────────────────────
# MODELS
# ────────────────────────────────────────────────

def get_models():

    return {
        "MLP":
            MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                          max_iter=300,
                          early_stopping=True),

        "AdaBoost":
            AdaBoostClassifier(n_estimators=100),

        "DNN": "DNN",
        "CNN": "CNN",
        "LSTM": "LSTM",
    }


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

def main():

    results = []

    for ds in DATASET_FILES:

        print("\n" + "=" * 70)
        print("Dataset:", ds["name"])

        X_train, X_test, y_train, y_test = load_dataset(ds)

        models = get_models()


        for name, model in models.items():

            print("Training:", name)

            start = time.time()

            if name in ["DNN", "CNN", "LSTM"]:

                if name == "DNN":
                    model_obj, y_pred, y_score = train_dnn(X_train, y_train, X_test)

                elif name == "CNN":
                    model_obj, y_pred, y_score = train_cnn(X_train, y_train, X_test)

                else:
                    model_obj, y_pred, y_score = train_lstm(X_train, y_train, X_test)

                model_obj.save(MODEL_DIR / f"{ds['name']}_{name}.h5")

            else:

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                y_score = model.predict_proba(X_test)

                with open(MODEL_DIR / f"{ds['name']}_{name}.pkl", "wb") as f:
                    pickle.dump(model, f)

            train_time = time.time() - start

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro")
            rec = recall_score(y_test, y_pred, average="macro")

            FAR, FAP, FAF1, DR, FPR = compute_ids_metrics(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)

            plot_confusion_matrix(cm, ds["name"], name)

            if y_score is not None:
                plot_roc(y_test, y_score, ds["name"], name)

            results.append({
                "Dataset": ds["name"],
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

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    df.to_excel("results/model_results.xlsx", index=False)

    print("\nDone! Models saved in saved_models/")


if __name__ == "__main__":
    main()