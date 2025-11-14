# fill_parquet_nans_to_dataset_test.py
import pandas as pd
import os

# Input parquet (your original)
PARQUET_PATH = r"D:\VS Code\CV\American Sign Language Translator\dataset\train_landmark_files\29302\293142.parquet"

# Output directory (dataset_test) and output file path
OUT_DIR = r"D:\VS Code\CV\American Sign Language Translator\dataset_test"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, os.path.basename(PARQUET_PATH))  # keeps same filename

if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"Input parquet not found: {PARQUET_PATH}")

print("Loading:", PARQUET_PATH)
df = pd.read_parquet(PARQUET_PATH)

# Show NaN counts before
nan_before = df.isna().sum()
print("NaN counts before (showing non-zero only):")
print(nan_before[nan_before > 0])

# Fill only numeric columns (recommended)
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(0)

# If you also want to fill object/string columns (not recommended), uncomment:
# df = df.fillna(0)

# Show NaN counts after
nan_after = df.isna().sum()
print("NaN counts after (should be zero for numeric cols):")
print(nan_after[nan_after > 0])

# Save to dataset_test
print("Saving filled parquet to:", OUT_PATH)
df.to_parquet(OUT_PATH, index=False)
print("Done.")
