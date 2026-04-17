import pandas as pd
import numpy as np
import joblib
import os
import glob
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ======================
# PATH
# ======================
BASE_DATA_PATH = r'D:\luanvsn\.venv\final\data'
OUTPUT_DIR = os.path.join(BASE_DATA_PATH, 'preprocessed')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Thư mục lưu:", OUTPUT_DIR)

# ======================
# CONFIG DATASET
# ======================
DATASETS = {
    'cic17': {
        'folder_name': 'cicids-17',
        'label_column_patterns': ['label', 'Label'],
        'benign_keywords': ['BENIGN', 'Normal', 'normal', 'benign'],
        'drop_early_columns': [
            'flow id', 'source ip', 'destination ip', 'timestamp'
        ],
        'encoding': 'latin1',
        'no_header': False,
    },
    'nsl-kdd': {
        'folder_name': 'nsl-kdd',
        'benign_keywords': ['normal'],
        'encoding': 'utf-8',
        'no_header': True,
    },
    'signal22': {
        'folder_name': 'Simargl2022',
        'label_column_patterns': ['anomaly'],
        'benign_keywords': ['normal','benign','good','non-attack','no attack','0','false','0.0'],
        'encoding': 'utf-8',
        'no_header': False,
    }
}

K_FEATURES = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ======================
# MAIN LOOP
# ======================
for DATASET_KEY, config in DATASETS.items():

    print("\n" + "="*80)
    print("DATASET:", DATASET_KEY.upper())
    print("="*80)

    DATA_FOLDER = os.path.join(BASE_DATA_PATH, config['folder_name'])
    OUTPUT_PREFIX = f"preprocessed_{DATASET_KEY}"

    if not os.path.exists(DATA_FOLDER):
        print("❌ Không tồn tại:", DATA_FOLDER)
        continue

    files = glob.glob(os.path.join(DATA_FOLDER, '*.csv')) + \
            glob.glob(os.path.join(DATA_FOLDER, '*.txt'))

    if not files:
        print("❌ Không có file")
        continue

    print("Found files:", len(files))

    dfs = []

    # ======================
    # LOAD FILE
    # ======================
    for f in files:
        try:
            df_temp = pd.read_csv(
                f,
                encoding=config['encoding'],
                engine='python',
                on_bad_lines='skip',
                header=None if config.get('no_header') else 0
            )

            if not config.get('no_header'):
                df_temp.columns = df_temp.columns.str.strip().str.lower()

            print(f"Loaded: {os.path.basename(f)} ({df_temp.shape})")
            dfs.append(df_temp)

        except Exception as e:
            print("❌ Lỗi:", e)

    if not dfs:
        continue

    df = pd.concat(dfs, ignore_index=True)
    print("Total rows:", len(df))

    # ======================
    # NSL-KDD FIX
    # ======================
    if DATASET_KEY == 'nsl-kdd':
        nsl_columns = [
            'duration','protocol_type','service','flag','src_bytes','dst_bytes',
            'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
            'num_compromised','root_shell','su_attempted','num_root',
            'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
            'is_host_login','is_guest_login','count','srv_count','serror_rate',
            'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
            'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
            'dst_host_same_srv_rate','dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
            'dst_host_serror_rate','dst_host_srv_serror_rate',
            'dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
        ]

        if len(df.columns) >= 42:
            df = df.iloc[:, :42]
            df.columns = nsl_columns

    # ======================
    # DROP METADATA
    # ======================
    drop_cols = config.get('drop_early_columns', [])
    drop_cols = [c for c in drop_cols if c in df.columns]

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print("Dropped metadata:", drop_cols)

    # ======================
    # CLEAN DATA
    # ======================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # ======================
    # FIND LABEL
    # ======================
    label_col = None

    if DATASET_KEY == 'signal22':
        label_col = 'anomaly' if 'anomaly' in df.columns else None

    elif DATASET_KEY == 'nsl-kdd':
        label_col = 'label'

    else:
        for pat in config['label_column_patterns']:
            cols = [c for c in df.columns if pat.lower() in c]
            if cols:
                label_col = cols[0]
                break

    if label_col is None:
        print("❌ Không tìm thấy label")
        continue

    print("Label:", label_col)

    # ======================
    # BINARY LABEL
    # ======================
    label_values = df[label_col].astype(str).str.lower().str.strip()
    benign_mask = label_values.isin([kw.lower() for kw in config['benign_keywords']])

    df['binary_label'] = (~benign_mask).astype(np.int8)
    df.drop(columns=[label_col], inplace=True)

    # ======================
    # 🔥 FIX: ENCODE CATEGORICAL (KHÔNG DROP)
    # ======================
    df = pd.get_dummies(df, drop_first=True)

    # ======================
    # SPLIT X, y
    # ======================
    X = df.drop(columns=['binary_label'])
    y = df['binary_label'].values

    print("X shape:", X.shape)

    # ======================
    # SCALE
    # ======================
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================
    # FEATURE SELECTION
    # ======================
    k_actual = min(K_FEATURES, X.shape[1])

    selector = SelectKBest(chi2, k=k_actual)
    X_selected = selector.fit_transform(X_scaled, y)

    selected_features = X.columns[selector.get_support()].tolist()

    print("Selected features:", len(selected_features))

    # 🔥 GIỮ TÊN
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

    # ======================
    # SPLIT TRAIN TEST
    # ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected_df, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ======================
    # SMOTE
    # ======================
    smote = SMOTE(random_state=RANDOM_STATE)

    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 🔥 giữ tên sau SMOTE
    X_train_res = pd.DataFrame(X_train_res, columns=selected_features)

    # ======================
    # SAVE
    # ======================
    base_path = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX)

    np.save(f'{base_path}_X_train.npy', X_train_res.values.astype(np.float32))
    np.save(f'{base_path}_X_test.npy', X_test.values.astype(np.float32))
    np.save(f'{base_path}_y_train.npy', y_train_res.astype(np.int8))
    np.save(f'{base_path}_y_test.npy', y_test.astype(np.int8))

    # 🔥 QUAN TRỌNG NHẤT
    joblib.dump(selected_features, f'{base_path}_feature_names.pkl')

    joblib.dump(scaler, f'{base_path}_scaler.pkl')
    joblib.dump(selector, f'{base_path}_selector.pkl')

    print(f"\n✅ DONE: {DATASET_KEY.upper()}")
    print("Saved to:", OUTPUT_DIR)

print("\n🔥 HOÀN TẤT TOÀN BỘ!")