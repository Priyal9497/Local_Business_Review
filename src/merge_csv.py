
import pandas as pd
import glob

files = [
    "Data/raw/kanika.csv",
    "Data/raw/Priyal restaurant reviews.csv",
    "Data/processed/RESTAURANT REVIEW FAIZAN_FIXED.csv",
    "Data/processed/Rishabh restaurant reviews_FIXED.csv"
]
dfs = []

for file in files:
    print("Reading:", file)
    df = pd.read_csv(file, encoding="latin1")
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

print("Merged Shape:", merged_df.shape)
print("Columns:", merged_df.columns.tolist())

merged_df.to_csv(
    "Data/processed/merged_restaurant_reviews.csv",
    index=False,
    encoding="utf-8"
)

print("✅ Dataset merged successfully!")