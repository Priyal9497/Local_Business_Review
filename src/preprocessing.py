import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load cleaned dataset
df = pd.read_csv("Data/processed/cleaned_restaurant_reviews.csv")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Reviews:")
print(df["Review text"].head())
# Tokenization
df["Tokens"] = df["Review text"].apply(lambda x: x.split())

print("\nFirst Tokenized Review:")
print(df["Tokens"].head())
# Remove stopwords
stop_words = set(stopwords.words("english"))

df["Tokens"] = df["Tokens"].apply(
    lambda words: [word for word in words if word.lower() not in stop_words]
)

print("\nAfter Removing Stopwords:")
print(df["Tokens"].head())

# Lemmatization
lemmatizer = WordNetLemmatizer()

df["Tokens"] = df["Tokens"].apply(
    lambda words: [lemmatizer.lemmatize(word) for word in words]
)

print("\nAfter Lemmatization:")
print(df["Tokens"].head())

# Save final preprocessed dataset
df.to_csv("Data/processed/preprocessed_restaurant_reviews.csv",index=False)

print("\nPreprocessed dataset saved successfully!")