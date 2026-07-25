import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("merged_restaurant_reviews.csv", encoding="utf-8")

# Basic information
print("\n===== DATASET OVERVIEW =====")
print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nReviews per Business:")
print(df["Business_Name"].value_counts())

print("\nRating Distribution:")
print(df["Rating"].value_counts().sort_index())
# Rating Distribution Bar Chart

rating_counts = df["Rating"].value_counts().sort_index()

plt.figure(figsize=(6,4))
rating_counts.plot(kind="bar")

plt.title("Rating Distribution")
plt.xlabel("Ratings")
plt.ylabel("Number of Reviews")

plt.savefig("rating_distribution.png")
plt.show()
# Reviews per Business

business_counts = df["Business_Name"].value_counts()

plt.figure(figsize=(8,5))
business_counts.plot(kind="bar")

plt.title("Number of Reviews per Business")
plt.xlabel("Business Name")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("reviews_per_business.png")
plt.show()
# Review Length Distribution

df["Review_Length"] = df["Review text"].astype(str).apply(len)

plt.figure(figsize=(8,5))
plt.hist(df["Review_Length"], bins=20)

plt.title("Distribution of Review Length")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Reviews")

plt.savefig("review_length_distribution.png")
plt.show()
# Word Cloud

text = " ".join(df["Review text"].astype(str))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Reviews")

plt.savefig("wordcloud.png")
plt.show()