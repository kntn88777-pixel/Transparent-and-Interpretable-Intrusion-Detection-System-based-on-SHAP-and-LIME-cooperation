import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path

# ==========================
# DATA DIR
# ==========================

DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")

# ==========================
# DATASET FILE MAP
# ==========================

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
# FEATURE NAMES
# ==========================

CIC17_FEATURES = [
"Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
"Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max",
"Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
"Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean",
"Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean",
"Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean",
"Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean",
"Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags",
"Fwd URG Flags","Bwd URG Flags","Fwd Header Length","Bwd Header Length",
"Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length",
"Packet Length Mean","Packet Length Std","Packet Length Variance",
"FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count",
"ACK Flag Count","URG Flag Count","CWE Flag Count","ECE Flag Count",
"Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size",
"Avg Bwd Segment Size","Subflow Fwd Packets","Subflow Fwd Bytes",
"Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward",
"Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward",
"Active Mean","Active Std","Active Max","Active Min","Idle Mean",
"Idle Std","Idle Max","Idle Min"
]

NSLKDD_FEATURES = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes",
"land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
"num_compromised","root_shell","su_attempted","num_root",
"num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_host_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
"diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
"dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

SIGNAL22_FEATURES = [
"FLOW_ID","PROTOCOL_MAP","L4_SRC_PORT","IPV4_SRC_ADDR","L4_DST_PORT",
"IPV4_DST_ADDR","FIRST_SWITCHED","FLOW_DURATION_MILLISECONDS",
"LAST_SWITCHED","PROTOCOL","TCP_FLAGS","TCP_WIN_MAX_IN",
"TCP_WIN_MAX_OUT","TCP_WIN_MIN_IN","TCP_WIN_MIN_OUT",
"TCP_WIN_MSS_IN","TCP_WIN_SCALE_IN","TCP_WIN_SCALE_OUT",
"SRC_TOS","DST_TOS","TOTAL_FLOWS_EXP","MIN_IP_PKT_LEN",
"MAX_IP_PKT_LEN","TOTAL_PKTS_EXP","TOTAL_BYTES_EXP",
"IN_BYTES","IN_PKTS","OUT_BYTES","OUT_PKTS","ALERT",
"ANALYSIS_TIMESTAMP"
]

FEATURE_NAMES = {
    "CIC17": CIC17_FEATURES,
    "NSL-KDD": NSLKDD_FEATURES,
    "SIGNAL2022": SIGNAL22_FEATURES
}

# ==========================
# LOAD DATA
# ==========================

def load_dataset(info):

    name = info["name"]

    X_train = np.load(DATA_DIR / info["X_train"], allow_pickle=True)
    X_test = np.load(DATA_DIR / info["X_test"], allow_pickle=True)
    Y_train = np.load(DATA_DIR / info["Y_train"], allow_pickle=True)
    Y_test = np.load(DATA_DIR / info["Y_test"], allow_pickle=True)

    print("\n========================")
    print("DATASET:", name)
    print("========================")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    columns = [f"feat_{i+1}" for i in range(X_train.shape[1])]

    X_train = pd.DataFrame(X_train, columns=columns)

    return X_train, Y_train


# ==========================
# FEATURE SELECTION
# ==========================

def select_features(X_train, Y_train):

    if Y_train.ndim > 1:
        Y_train = Y_train[:,0]

    selector = SelectKBest(f_classif, k='all')
    selector.fit(X_train, Y_train)

    scores = selector.scores_

    df = pd.DataFrame({
        "feature": X_train.columns,
        "score": scores
    })

    df = df.sort_values("score", ascending=False)

    return df


# ==========================
# VISUALIZATION
# ==========================

def plot_bar(df, dataset, top_k):

    top = df.head(top_k)

    plt.figure(figsize=(10,6))

    plt.barh(top["feature"], top["score"])

    plt.gca().invert_yaxis()

    plt.title(f"Top {top_k} Feature Importance ({dataset})")

    plt.tight_layout()

    plt.savefig(f"{dataset}_bar.png", dpi=300)

    plt.close()


def plot_heatmap(df, dataset):

    plt.figure(figsize=(8,6))

    sns.heatmap(df.head(20)[["score"]], annot=True, cmap="viridis")

    plt.title(f"Feature Heatmap ({dataset})")

    plt.tight_layout()

    plt.savefig(f"{dataset}_heatmap.png", dpi=300)

    plt.close()


def plot_table(df, dataset, top_k):

    top = df.head(top_k)

    names = FEATURE_NAMES.get(dataset)

    real = []

    for f in top["feature"]:

        idx = int(f.split("_")[1]) - 1

        if names and idx < len(names):
            real.append(names[idx])
        else:
            real.append(f)

    table_df = pd.DataFrame({

        "Rank": range(1, top_k+1),
        "Feature_ID": top["feature"],
        "Real_Name": real,
        "Score": np.round(top["score"],2)

    })

    fig, ax = plt.subplots(figsize=(12, top_k*0.5))

    ax.axis('off')

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1,1.5)

    plt.title(f"Feature Ranking ({dataset})")

    plt.savefig(f"{dataset}_feature_table.png", dpi=300, bbox_inches="tight")

    plt.close()


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    TOP_K = 20

    for i, dataset_info in enumerate(DATASET_FILES):

        print("\nProcessing dataset index:", i)

        X_train, Y_train = load_dataset(dataset_info)

        df_scores = select_features(X_train, Y_train)

        name = dataset_info["name"]

        plot_bar(df_scores, name, TOP_K)

        plot_heatmap(df_scores, name)

        plot_table(df_scores, name, TOP_K)

        print("Finished:", name)