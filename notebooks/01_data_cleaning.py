import pandas as pd

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

print("\nCleaned dataset saved successfully!")