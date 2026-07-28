import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from text_utils import (CLEAN_CSV, MODELS_DIR, clean_review_text, oov_ratio, probs_to_outputs, rating_to_sentiment, tag_aspects, LexiconFeatures, POS_WORDS, NEG_WORDS, NEGATORS)  

st.set_page_config(page_title="Restaurant Review Analyzer", layout="wide")

@st.cache_data
def load_data():
    d = pd.read_csv(CLEAN_CSV)
    d["Business_Name"] = d["Business_Name"].fillna("Unlisted").astype(str).str.strip()  # ← the fix
    d["sentiment_label"] = d["Rating"].apply(rating_to_sentiment)
    if "aspect" not in d.columns:
        d["aspect"] = d["Review text"].astype(str).apply(lambda t: tag_aspects(t)[0])
    return d

@st.cache_resource
def load_model():
    b = joblib.load(os.path.join(MODELS_DIR, "review_model.pkl"))
    return b["pipeline"], np.asarray(b["classes"])

df = load_data()
pipeline, CLASSES = load_model()
word_vec = pipeline.named_steps["feats"].transformer_list[0][1]

st.title("Restaurant Review Analyzer")
st.caption(f"Built on {len(df)} cleaned reviews across {df['Business_Name'].nunique()} restaurants")

business = st.selectbox("Select a restaurant", ["All"] + sorted(df["Business_Name"].unique()))
subset = df if business == "All" else df[df["Business_Name"] == business]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", len(subset))
c2.metric("Avg. Rating", round(subset["Rating"].mean(), 2) if len(subset) else 0)
c3.metric("% Positive",
          f"{(subset['sentiment_label'] == 'positive').mean() * 100:.1f}%" if len(subset) else "0%")
top_aspect = subset["aspect"].mode()
c4.metric("Top aspect", top_aspect[0] if len(subset) and not top_aspect.empty else "-")

a, b = st.columns(2)
with a:
    st.subheader("Rating Distribution")
    if len(subset):
        st.bar_chart(subset["Rating"].value_counts().sort_index())
with b:
    st.subheader("Reviews by Aspect")
    if len(subset):
        st.bar_chart(subset["aspect"].value_counts())

st.subheader("Average Rating by Restaurant")
st.bar_chart(df.groupby("Business_Name")["Rating"].mean().sort_values())

st.subheader("Word Cloud")
text = " ".join(subset["Review text"].astype(str))
if text.strip():
    wc = WordCloud(width=800, height=350, background_color="white", stopwords=set(STOPWORDS)).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ------------------------------------------------------------
st.subheader("Try it: predict sentiment, rating & topic from your own review")
user_text = st.text_area("Enter a review", placeholder="e.g. The biryani was amazing but the service was slow")

if st.button("Analyze") and user_text.strip():
    if len(user_text.split()) < 6:
        st.info("Short review — prediction is based on very little text.")

    cleaned = clean_review_text(user_text)
    proba = pipeline.predict_proba([cleaned])[0]
    rating, sentiment, sent_probs = probs_to_outputs(proba, CLASSES)
    aspect, all_aspects, _ = tag_aspects(user_text)
    oov = oov_ratio(user_text, word_vec)

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted sentiment", sentiment)
    k2.metric("Predicted rating", f"{rating:.1f}")
    k3.metric("Predicted topic", aspect)

    st.caption("Sentiment confidence: " +
               ", ".join(f"{k}: {v:.0%}" for k, v in sent_probs.items()))
    st.caption("Star distribution: " +
               ", ".join(f"{int(c)}★ {p:.0%}" for c, p in zip(CLASSES, proba)))
    if all_aspects:
        st.caption("Aspects mentioned: " + ", ".join(all_aspects))

    ordered = sorted(sent_probs.values(), reverse=True)
    if oov > 0.6:
        st.warning(f"{oov:.0%} of the words here were not seen during training — "
                   "the prediction relies mostly on general polarity cues.")
    if ordered[0] - ordered[1] < 0.15:
        st.warning("Sentiment classes were close — treat this as uncertain.")
    toks = set(cleaned.split())
    has_pos = bool(toks & POS_WORDS)
    has_neg = bool(toks & NEG_WORDS) or bool(toks & NEGATORS)
    if len(all_aspects) > 1 and has_pos and has_neg:
        st.info(f"Mixed review — mentions {' and '.join(all_aspects)} with both "
                "positive and negative points. The single score averages them.")
    elif len(all_aspects) > 1:
        st.info(f"Covers several aspects: {' and '.join(all_aspects)}.")