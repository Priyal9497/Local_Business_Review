import os
import sys
import re
import pandas as pd

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
for _c in (_HERE, os.path.dirname(_HERE)):
    _p = os.path.join(_c, "app")
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break

from text_utils import MERGED_CSV, CLEAN_CSV, clean_review_text

df = pd.read_csv(MERGED_CSV)
print("loaded:", df.shape)

df = df.dropna(subset=["Review text", "Rating"])
for col in ["Business_Name", "Location", "Review text"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

df["Review text"] = df["Review text"].apply(
    lambda x: re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s']", " ", str(x))).strip()
)

df = df.drop_duplicates(subset=["Review text"])
df = df[df["Review text"].str.split().str.len() >= 3]

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])
df["Rating"] = df["Rating"].round().clip(1, 5).astype(int)

df["review_clean"] = df["Review text"].apply(clean_review_text)

os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
df.to_csv(CLEAN_CSV, index=False)

print("saved:", CLEAN_CSV)
print(df["Rating"].value_counts().sort_index())