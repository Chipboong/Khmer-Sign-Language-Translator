import pandas as pd

# Path to your parquet file
file_path = r"D:\VS Code\CV\American Sign Language Translator\dataset\train_landmark_files\2044\635217.parquet"

# Read parquet file
# 


df = pd.read_parquet(file_path)

num_frames = df["frame"].nunique()

print("Total frames:", num_frames)
print(sorted(df["frame"].unique()))