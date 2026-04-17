import pandas as pd
import numpy as np
import joblib
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

BASE_PATH = r"D:\luanvsn\.venv\final\data\Simargl2022"
OUTPUT_PATH = r"D:\luanvsn\.venv\final\data\preprocessed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

files = glob.glob(os.path.join(BASE_PATH, "*.csv"))

dfs = []

# ======================
# LOAD + LABEL + SAMPLE
# ======================
for f in files:
    name = os.path.basename(f).lower()
    print("Loading", name)

    df = pd.read_csv(f)
    df.columns = df.columns.str.lower()

    # ===== gán label =====
    if "normal" in name:
        df["binary_label"] = 0
    else:
        df["binary_label"] = 1

    # ===== giảm RAM =====
    df = df.sample(min(len(df), 200000), random_state=42)

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print("Rows after sampling:", len(df))

# ======================
# DROP METADATA
# ======================
drop_cols = [
    "flow_id",
    "ipv4_src_addr",
    "ipv4_dst_addr",
    "analysis_timestamp",
    "first_switched",
    "last_switched"
]

df.drop(columns=drop_cols, inplace=True, errors="ignore")

# ======================
# CLEAN DATA
# ======================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ======================
# SPLIT X y
# ======================
X = df.drop(columns=["binary_label"])
y = df["binary_label"]

# ===== FIX STRING =====
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print("Dropping string columns:", object_cols)
    X = X.drop(columns=object_cols)

print("X shape:", X.shape)

# ======================
# 🚨 SPLIT TRƯỚC (QUAN TRỌNG)
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ======================
# SCALE (FIT TRÊN TRAIN)
# ======================
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# FEATURE SELECTION
# ======================
selector = SelectKBest(chi2, k=24)

X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print("Selected features:", list(selected_features))

# ======================
# SMOTE (CHỈ TRAIN)
# ======================
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", X_train.shape)

# ======================
# SAVE
# ======================
np.save(os.path.join(OUTPUT_PATH, "signal22_X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "signal22_X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_PATH, "signal22_y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "signal22_y_test.npy"), y_test)

joblib.dump(scaler, os.path.join(OUTPUT_PATH, "signal22_scaler.pkl"))
joblib.dump(selector, os.path.join(OUTPUT_PATH, "signal22_selector.pkl"))

print("DONE SIGNAL2022 (NO LEAKAGE)")