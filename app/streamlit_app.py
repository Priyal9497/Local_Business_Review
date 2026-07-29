import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud, STOPWORDS

from text_utils import (CLEAN_CSV, MODELS_DIR, clean_review_text, oov_ratio, probs_to_outputs, rating_to_sentiment, tag_aspects, LexiconFeatures, POS_WORDS, NEG_WORDS, NEGATORS)  

st.set_page_config(page_title="Restaurant Review Analyzer", layout="wide")

st.markdown("""
<style>

/* Dashboard metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(30, 41, 59, 0.95),
        rgba(15, 23, 42, 0.95)
    );

    border: 1px solid rgba(255, 255, 255, 0.10);
    padding: 20px 22px;
    border-radius: 16px;

    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.20);

    transition:
        transform 0.2s ease,
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.55);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
}

/* Metric label */
div[data-testid="stMetricLabel"] {
    font-size: 15px;
    opacity: 0.8;
}

/* Metric value */
div[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

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

st.markdown(
    f"""
    <div style="padding:28px 32px; border-radius:18px; margin-bottom:24px; background:linear-gradient(135deg,#1f2937,#111827); border:1px solid rgba(255,255,255,0.10);">
        <h1 style="margin:0; font-size:42px;">🍽️ Restaurant Review Analyzer</h1>
        <p style="margin:10px 0 0 0; font-size:17px; opacity:0.8;">Discover customer sentiment, ratings and key insights from restaurant reviews.</p>
        <p style="margin:8px 0 0 0; font-size:14px; opacity:0.6;">Analyzing {len(df)} cleaned reviews across {df['Business_Name'].nunique()} restaurants</p>
    </div>
    """,
    unsafe_allow_html=True
)
business = st.selectbox("Select a restaurant", ["All"] + sorted(df["Business_Name"].unique()))
subset = df if business == "All" else df[df["Business_Name"] == business]

st.markdown("### Dashboard Overview")

review_count = len(subset)

avg_rating = round(subset["Rating"].mean(), 2) if len(subset) else 0

positive_pct = (
    f"{(subset['sentiment_label'] == 'positive').mean() * 100:.1f}%"
    if len(subset)
    else "0.0%"
)

top_aspect = subset["aspect"].mode()
top_aspect_value = (
    top_aspect.iloc[0]
    if len(subset) and not top_aspect.empty
    else "N/A"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("📝 Reviews", review_count)
c2.metric("⭐ Avg. Rating", avg_rating)
c3.metric("😊 Positive", positive_pct)
c4.metric("🍽️ Top Aspect", top_aspect_value)
a, b = st.columns(2)
with a:
    st.subheader("⭐ Rating Distribution")

    if len(subset):
        rating_data = (
            subset["Rating"]
            .value_counts()
            .sort_index()
            .rename_axis("Rating")
            .reset_index(name="Reviews")
        )

        rating_chart = (
            alt.Chart(rating_data)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Rating:O", title="Rating"),
                y=alt.Y("Reviews:Q", title="Number of Reviews"),
                tooltip=[
                    alt.Tooltip("Rating:O", title="Rating"),
                    alt.Tooltip("Reviews:Q", title="Reviews"),
                ],
            )
            .properties(height=320)
        )

        st.altair_chart(rating_chart, use_container_width=True)
with b:
    st.subheader("🍽️ Reviews by Aspect")

    if len(subset):
        aspect_data = (
            subset["aspect"]
            .value_counts()
            .rename_axis("Aspect")
            .reset_index(name="Reviews")
        )

        aspect_chart = (
            alt.Chart(aspect_data)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Aspect:N", title="Aspect", sort="-y"),
                y=alt.Y("Reviews:Q", title="Number of Reviews"),
                tooltip=[
                    alt.Tooltip("Aspect:N", title="Aspect"),
                    alt.Tooltip("Reviews:Q", title="Reviews"),
                ]
            )
            .properties(height=320)
        )

        st.altair_chart(aspect_chart, use_container_width=True)
st.subheader("⭐ Average Rating by Restaurant")

rating_by_restaurant = (
    df.groupby("Business_Name")["Rating"]
    .mean()
    .round(2)
    .reset_index()
)

restaurant_chart = (
    alt.Chart(rating_by_restaurant)
    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
    .encode(
        x=alt.X(
            "Rating:Q",
            title="Average Rating",
            scale=alt.Scale(domain=[0, 5])
        ),
        y=alt.Y(
            "Business_Name:N",
            title=None,
            sort="-x"
        ),
        tooltip=[
            alt.Tooltip("Business_Name:N", title="Restaurant"),
            alt.Tooltip("Rating:Q", title="Average Rating", format=".2f")
        ]
    )
    .properties(height=400)
)

st.altair_chart(restaurant_chart, use_container_width=True)
st.subheader("☁️ Word Cloud")

text = " ".join(subset["Review text"].astype(str))

if text.strip():
    wc = WordCloud(
        width=1200,
        height=450,
        background_color="#0E1117",
        colormap="Blues",
        stopwords=set(STOPWORDS),
        max_words=120
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
# ------------------------------------------------------------
st.markdown("## 🔍 Analyze Your Review")
st.caption("Enter a restaurant review and let the model predict its sentiment, rating, and topic.")
user_text = st.text_area(
    "✍️ Enter your review",
    placeholder="e.g. The biryani was amazing but the service was slow",
    height=130
)

if st.button("✨ Analyze Review", type="primary", use_container_width=True) and user_text.strip():

    if len(user_text.split()) < 6:
        st.info("Short review - prediction is based on very little text.")

    cleaned = clean_review_text(user_text)
    proba = pipeline.predict_proba([cleaned])[0]

    rating, sentiment, sent_probs = probs_to_outputs(proba, CLASSES)
    aspect, all_aspects, _ = tag_aspects(user_text)
    oov = oov_ratio(user_text, word_vec)

    st.markdown("### 📊 Analysis Results")

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted sentiment", sentiment)
    k2.metric("Predicted rating", f"{rating:.1f}")
    k3.metric("Predicted topic", aspect)

    st.markdown("#### 🔎 Prediction Details")

    st.markdown(
        "**Sentiment confidence:** " +
        " · ".join(
            f"{k.title()} **{v:.0%}**"
            for k, v in sent_probs.items()
        )
    )

    st.markdown(
        "**⭐ Star distribution:** " +
        " · ".join(
            f"{int(c)}★ **{p:.0%}**"
            for c, p in zip(CLASSES, proba)
        )
    )

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