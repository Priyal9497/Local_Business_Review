'''import pandas as pd

# Load dataset
df = pd.read_csv(
    "Data/processed/merged_restaurant_reviews.csv"
)

print("Shape of Dataset:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())
# Check duplicate rows
duplicates = df.duplicated().sum()

print("\nNumber of Duplicate Rows:", duplicates)

# Remove duplicates
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Remove rows with missing values
df = df.dropna()

print("\nShape after removing missing values:", df.shape)
# Remove leading and trailing spaces
df["Business_Name"] = df["Business_Name"].str.strip()
df["Location"] = df["Location"].str.strip()
df["Review text"] = df["Review text"].str.strip()

print("\nExtra spaces removed successfully!")
# Convert review text to lowercase
df["review_clean"] = df["Review text"].str.lower()

print("\nReview text converted to lowercase successfully!")
print(df.head())
import re

# Remove special characters
df["Review text"] = df["Review text"].apply(
    lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x)
)

# Remove extra spaces between words
df["Review text"] = df["Review text"].str.replace(r"\s+", " ", regex=True).str.strip()

print("\nSpecial characters removed successfully!")
# Save cleaned dataset
df.to_csv("Data/processed/cleaned_restaurant_reviews.csv", index=False)

print("\nCleaned dataset saved successfully!")'''


# ============================================================
# 01_clean_data.py
# FIXES:
#  - re.sub(...,"") -> " "   (was creating "burgera", "afan")
#  - review_clean regenerated LAST (was out of sync)
#  - drop_duplicates on Review text (was all-columns, missed 4)
#  - dropna only on required columns (was dropping rows for missing Location)
# ============================================================
# ============================================================
# notebook/01_data_cleaning.py — REPLACE ENTIRE FILE
# ============================================================
# ============================================================
# notebook/01_data_cleaning.py — REPLACE ENTIRE FILE
# FIXES: punctuation -> space (no more "burgera"), review_clean
# regenerated last with negation markers, dedupe on Review text.
# ============================================================
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

df["Business_Name"] = df["Business_Name"].replace({"Trufflles": "Truffles"})

# THE FIX: punctuation -> SPACE (old code used "" and created "burgera", "afan")
df["Review text"] = df["Review text"].apply(
    lambda x: re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s']", " ", str(x))).strip()
)

df = df.drop_duplicates(subset=["Review text"])
df = df[df["Review text"].str.split().str.len() >= 3]

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])
df["Rating"] = df["Rating"].round().clip(1, 5).astype(int)

# regenerated LAST so it includes every fix + negation markers
df["review_clean"] = df["Review text"].apply(clean_review_text)

os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
df.to_csv(CLEAN_CSV, index=False)

print("saved:", CLEAN_CSV)
print(df["Rating"].value_counts().sort_index())
print("\nexample review_clean:")
print(df["review_clean"].iloc[0][:200])