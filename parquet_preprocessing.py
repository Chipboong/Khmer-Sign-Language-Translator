import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# ======================
# CONFIG
# ======================
ROOT = "dataset"
LANDMARK_DIR = os.path.join(ROOT, "train_landmark_files")
OUTPUT_DIR = os.path.join(ROOT, "processed")
TARGET_FRAMES = 50          # you can change this

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# LOAD CSV
# ======================
train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
# train.csv must have: path , label

for idx, row in train_df.iterrows():
    rel_path = row["path"]          # e.g. "landmarks/data1/data.parquet"
    label = row["sign"]            # e.g. "A"

    parquet_path = os.path.join(ROOT, rel_path)

    if not os.path.exists(parquet_path):
        print("Missing file:", parquet_path)
        continue

    # ======================
    # READ PARQUET
    # ======================
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # convert DataFrame → NumPy
    data = df.to_numpy()

    # ======================
    # FIX BAD VALUES
    # ======================

    # 1) Replace NaN with 0
    data = np.nan_to_num(data, nan=0.0)


    # ======================
    # PAD / TRIM TO TARGET LENGTH
    # ======================
    frames = data.shape[0]

    if frames < TARGET_FRAMES:
        pad_len = TARGET_FRAMES - frames
        data = np.pad(data, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
    else:
        data = data[:TARGET_FRAMES]

    # ======================
    # SAVE OUTPUT
    # ======================
    label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    seq_name = os.path.basename(os.path.dirname(parquet_path))  # e.g. "data1"
    save_path = os.path.join(label_dir, f"{seq_name}.npy")

    np.save(save_path, data)

    print("Saved:", save_path)

print("Done preparing training data.")
