# ============================================================
# IMPORTS AND SETUP
# ============================================================
import os
import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary

from text_utils import clean_review_text, oov_ratio, get_dominant_topic

st.set_page_config(page_title="Bengaluru Restaurant Review Analyzer", layout="wide")

# ============================================================
# PATH SAFETY
# Build absolute paths from this script's own location, so the app
# works no matter what folder you run `streamlit run` from.
# Adjust the ".." below if your data/models folders live somewhere
# else relative to this file.
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # one level up from /app
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_restaurant_reviews.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ============================================================
# TOPIC LABELS
# Copy these from Step 6 of your topic modeling notebook once you've
# read the top words per topic and labeled them (replace the TODOs).
# ============================================================
TOPIC_LABELS = {
    0: "TODO: label topic 0",
    1: "TODO: label topic 1",
    2: "TODO: label topic 2",
    3: "TODO: label topic 3",
    4: "TODO: label topic 4",
}

# ============================================================
# CACHED LOADERS FOR DATA AND MODELS
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    clf = joblib.load(os.path.join(MODELS_DIR, "sentiment_classifier.pkl"))
    reg = joblib.load(os.path.join(MODELS_DIR, "rating_regressor.pkl"))
    lda_model = LdaModel.load(os.path.join(MODELS_DIR, "lda_model.gensim"))
    dictionary = Dictionary.load(os.path.join(MODELS_DIR, "lda_dictionary.gensim"))
    return tfidf, clf, reg, lda_model, dictionary

df = load_data()

# ============================================================
# DERIVE SENTIMENT LABELS FROM RATINGS
# ============================================================
def rating_to_sentiment(r):
    if r >= 4:
        return "positive"
    elif r == 3:
        return "neutral"
    return "negative"

df["sentiment_label"] = df["Rating"].apply(rating_to_sentiment)
tfidf, clf, reg, lda_model, dictionary = load_models()

# ============================================================
# CACHED: compute dominant topic for every review in the dataset
# (only runs once per data version, thanks to st.cache_data)
# ============================================================
@st.cache_data
def compute_topics(_lda_model, _dictionary, reviews):
    topic_ids = []
    for text in reviews:
        topic_id, _ = get_dominant_topic(text, _lda_model, _dictionary)
        topic_ids.append(topic_id)
    return topic_ids

df["dominant_topic"] = compute_topics(lda_model, dictionary, df["review_clean"].astype(str).tolist())
df["topic_label"] = df["dominant_topic"].map(lambda t: TOPIC_LABELS.get(t, "Unknown") if t is not None else "Unknown")

# ============================================================
# PAGE TITLE AND RESTAURANT FILTER
# ============================================================
st.title("Bengaluru Restaurant Review Analyzer")
st.caption(f"Built on {len(df)} cleaned reviews across {df['Business_Name'].nunique()} Bengaluru restaurants")

business_options = ["All"] + sorted(df["Business_Name"].unique().tolist())
business = st.selectbox("Select a restaurant", business_options)
subset = df if business == "All" else df[df["Business_Name"] == business]

# ============================================================
# SUMMARY METRICS
# ============================================================
col1, col2, col3 = st.columns(3)
col1.metric("Reviews", len(subset))
col2.metric("Avg. Rating", round(subset["Rating"].mean(), 2) if len(subset) else 0)
pct_pos = (subset["sentiment_label"] == "positive").mean() * 100 if len(subset) else 0
col3.metric("% Positive", f"{pct_pos:.1f}%")

# ============================================================
# RATING DISTRIBUTION CHART
# ============================================================
st.subheader("Rating Distribution")
if len(subset):
    st.bar_chart(subset["Rating"].value_counts().sort_index())

# ============================================================
# AVERAGE RATING BY RESTAURANT CHART
# ============================================================
st.subheader("Average Rating by Restaurant")
avg_by_biz = df.groupby("Business_Name")["Rating"].mean().sort_values()
st.bar_chart(avg_by_biz)

# ============================================================
# NEW: REVIEWS BY TOPIC CHART
# ============================================================
st.subheader("Reviews by Topic")
if len(subset):
    topic_counts = subset["topic_label"].value_counts()
    st.bar_chart(topic_counts)
else:
    st.caption("No reviews in this selection.")

# ============================================================
# WORD CLOUD
# ============================================================
st.subheader("Word Cloud")
text = " ".join(subset["review_clean"].astype(str))
if text.strip():
    wc = WordCloud(width=800, height=350, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ============================================================
# LIVE PREDICTION: SENTIMENT + RATING + TOPIC FROM USER REVIEW
# ============================================================
st.subheader("Try it: predict sentiment, rating & topic from your own review")
user_text = st.text_area("Enter a review", placeholder="e.g. The biryani was amazing but the service was slow")

if st.button("Analyze") and user_text.strip():
    if len(user_text.split()) < 6:
        st.warning("Very short reviews give unreliable predictions — try a full sentence.")

    # ---- Sentiment + rating ----
    cleaned = clean_review_text(user_text)
    vec = tfidf.transform([cleaned])
    oov = oov_ratio(cleaned, tfidf)

    pred_sentiment = clf.predict(vec)[0]
    raw_rating = float(reg.predict(vec)[0])
    pred_rating = min(5.0, max(1.0, round(raw_rating, 1)))  # clip to valid 1-5 range

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vec)[0]
        proba_map = dict(zip(clf.classes_, proba))
        top_conf = max(proba)
    else:
        proba_map = None
        top_conf = None

    # ---- Topic ----
    topic_id, topic_prob = get_dominant_topic(user_text, lda_model, dictionary)
    topic_label = TOPIC_LABELS.get(topic_id, "Unknown") if topic_id is not None else "Not enough text to infer a topic"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted sentiment", pred_sentiment)
    col_b.metric("Predicted rating", pred_rating)
    col_c.metric("Predicted topic", topic_label)

    if proba_map is not None:
        st.caption("Sentiment confidence: " + ", ".join(f"{k}: {v:.0%}" for k, v in proba_map.items()))
    if topic_prob is not None:
        st.caption(f"Topic confidence: {topic_prob:.0%}")

    if oov > 0.5:
        st.warning(
            f"⚠️ {oov:.0%} of the words/phrases in this review were not seen "
            "during training, so the sentiment/rating prediction is based on "
            "very little real signal — treat it as unreliable."
        )
    if top_conf is not None and top_conf < 0.6:
        st.warning("⚠️ The sentiment model isn't confident — the classes were close, close to a coin-flip.")