import time
import numpy as np
import pandas as pd
import os
import pickle

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ==========================
# DATA PATH
# ==========================

DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")


DATASET_FILES = [

    {
        "name": "CIC17",
        "X_train": "preprocessed_cic17_X_train.npy",
        "X_test": "preprocessed_cic17_X_test.npy",
        "Y_train": "preprocessed_cic17_y_train.npy",
        "Y_test": "preprocessed_cic17_y_test.npy"
    },

    {
        "name": "NSL-KDD",
        "X_train": "preprocessed_nsl-kdd_X_train.npy",
        "X_test": "preprocessed_nsl-kdd_X_test.npy",
        "Y_train": "preprocessed_nsl-kdd_y_train.npy",
        "Y_test": "preprocessed_nsl-kdd_y_test.npy"
    },

    {
        "name": "SIGNAL2022",
        "X_train": "signal22_X_train.npy",
        "X_test": "signal22_X_test.npy",
        "Y_train": "signal22_y_train.npy",
        "Y_test": "signal22_y_test.npy"
    }

]


# ==========================
# METRICS
# ==========================

def compute_ids_metrics(y_true, y_pred):

    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin)

    if cm.shape == (2,2):

        TN, FP, FN, TP = cm.ravel()

        FAR = FP/(FP+TN) if (FP+TN)!=0 else 0

        FAP = FP/(FP+TP) if (FP+TP)!=0 else 0

        FAF1 = (2*TP)/(2*TP + FN + 2*FP) if (2*TP+FN+2*FP)!=0 else 0

        DR = TP/(TP+FN) if (TP+FN)!=0 else 0

        FPR = FP/(FP+TN) if (FP+TN)!=0 else 0

    else:

        FAR = FAP = FAF1 = DR = FPR = 0

    return FAR, FAP, FAF1, DR, FPR


# ==========================
# LOAD DATA
# ==========================

def load_dataset(ds):

    print("\nLoading:", ds["name"])

    X_train = np.load(DATA_DIR / ds["X_train"])
    X_test  = np.load(DATA_DIR / ds["X_test"])

    y_train = np.load(DATA_DIR / ds["Y_train"])
    y_test  = np.load(DATA_DIR / ds["Y_test"])

    if len(y_train.shape) > 1:

        y_train = np.argmax(y_train, axis=1)
        y_test  = np.argmax(y_test, axis=1)

    return X_train, X_test, y_train, y_test


# ==========================
# PLOT CONFUSION MATRIX
# ==========================

def plot_confusion_matrix(cm, dataset, model):

    os.makedirs("plots/confusion", exist_ok=True)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{dataset} - {model} Confusion Matrix")

    plt.ylabel("True")

    plt.xlabel("Predicted")

    plt.savefig(f"plots/confusion/{dataset}_{model}_cm.png")

    plt.close()


# ==========================
# ROC CURVE
# ==========================

def plot_roc(y_true, y_score, dataset, model):

    os.makedirs("plots/roc", exist_ok=True)

    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())

    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

    plt.plot([0,1],[0,1],'--')

    plt.title(f"{dataset} - {model} ROC")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.savefig(f"plots/roc/{dataset}_{model}_roc.png")

    plt.close()


# ==========================
# DNN
# ==========================

def train_dnn(X_train, y_train, X_test):

    num_classes = len(np.unique(y_train))

    model = Sequential([

        Dense(128, activation='relu', input_dim=X_train.shape[1]),

        Dense(64, activation='relu'),

        Dense(num_classes, activation='softmax')

    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=0)

    pred = model.predict(X_test)

    return np.argmax(pred, axis=1), pred


# ==========================
# MODELS
# ==========================

def get_models():

    return {

        "RandomForest": RandomForestClassifier(n_estimators=200,max_depth=15,n_jobs=-1),

        "KNN": KNeighborsClassifier(n_neighbors=10,n_jobs=-1),

        "MLP": MLPClassifier(hidden_layer_sizes=(128,64),max_iter=200),

        "AdaBoost": AdaBoostClassifier(),

        "SGD_SVM": SGDClassifier(loss="hinge"),

        "DNN": "DNN"
    }


# ==========================
# MAIN
# ==========================

def main():

    results = []

    models = get_models()

    for ds in DATASET_FILES:

        X_train, X_test, y_train, y_test = load_dataset(ds)

        for name, model in models.items():

            print(f"\nTraining {name} on {ds['name']}")

            start = time.time()

            if name == "DNN":

                y_pred, y_score = train_dnn(X_train,y_train,X_test)

            else:

                model.fit(X_train,y_train)

                y_pred = model.predict(X_test)

                if hasattr(model,"predict_proba"):

                    y_score = model.predict_proba(X_test)

                else:

                    y_score = np.zeros((len(y_pred),1))

            train_time = time.time()-start


            acc = accuracy_score(y_test,y_pred)

            prec = precision_score(y_test,y_pred,average="macro",zero_division=0)

            rec = recall_score(y_test,y_pred,average="macro",zero_division=0)

            FAR,FAP,FAF1,DR,FPR = compute_ids_metrics(y_test,y_pred)


            cm = confusion_matrix(y_test,y_pred)

            plot_confusion_matrix(cm,ds["name"],name)

            try:

                plot_roc(y_test,y_score,ds["name"],name)

            except:

                pass


            results.append({

                "Dataset":ds["name"],
                "Model":name,

                "Accuracy":acc,
                "Precision":prec,
                "Recall":rec,

                "DetectionRate":DR,
                "FPR":FPR,

                "FAF1":FAF1,
                "FAR":FAR,
                "FAP":FAP,

                "TrainTime":train_time
            })


    df = pd.DataFrame(results)

    os.makedirs("results",exist_ok=True)

    df.to_csv("results/model_results.csv",index=False)

    df.to_excel("results/model_results.xlsx",index=False)


    # ==========================
    # MODEL COMPARISON PLOT
    # ==========================

    os.makedirs("plots/comparison",exist_ok=True)

    for dataset in df["Dataset"].unique():

        subset = df[df["Dataset"]==dataset]

        plt.figure(figsize=(8,5))

        sns.barplot(data=subset,x="Model",y="Accuracy")

        plt.title(f"{dataset} Model Accuracy Comparison")

        plt.xticks(rotation=30)

        plt.savefig(f"plots/comparison/{dataset}_accuracy.png")

        plt.close()


    print("\nResults saved to results/model_results.xlsx")


# ==========================

if __name__ == "__main__":

    main()