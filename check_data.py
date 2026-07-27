
import pandas as pd, numpy as np
df = pd.read_csv("Data/processed/cleaned_restaurant_reviews.csv")         # ← your actual filename




print("--- derived sentiment (if rating-based) ---")
s = pd.cut(df.Rating, [0,2.5,3.5,5], labels=["negative","neutral","positive"])
print(s.value_counts())

print("\n--- text length ---")
print(df["Review text"].str.split().str.len().describe())

print("\n--- duplicates ---")
print("exact dupes:", df["Review text"].duplicated().sum())

print("\n--- sample reviews ---")
for t in df["Review text"].sample(5, random_state=0):
    print("-", t[:200])