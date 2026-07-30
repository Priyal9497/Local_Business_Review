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
/* Different colors for Dashboard Overview cards */

/* Reviews */
div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #2563EB, #06B6D4) !important;
}

/* Avg. Rating */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #7C3AED, #C084FC) !important;
}

/* Positive */
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #059669, #34D399) !important;
}

/* Top Aspect */
div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #EA580C, #FBBF24) !important;
}
/* =========================
   SIDEBAR DESIGN
   ========================= */

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #172033 50%,
        #1e293b 100%
    );
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Sidebar inner spacing */
section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

/* Sidebar text */
section[data-testid="stSidebar"] label {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

/* Select boxes */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-radius: 10px;
    border: 1px solid #cbd5e1;
    color: #111827;
}

/* Multiselect selected tags */
section[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background: linear-gradient(135deg, #ff5f6d, #ff7a59);
    color: white;
    border-radius: 8px;
    font-weight: 600;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff;
    font-weight: 700;
}
/* Sidebar footer */
.sidebar-footer {
    margin-top: 70px;
    padding: 18px 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.8;
}

.sidebar-footer .team-names {
    color: #f8fafc;
    font-weight: 600;
}

.sidebar-footer .project-name {
    margin-top: 6px;
    color: #94a3b8;
    font-size: 11px;
}
/* ==================================
   COLORFUL DASHBOARD METRIC CARDS
   ================================== */

/* Reviews - Blue */
div[data-testid="stMetric"]:nth-of-type(1) {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    border: 1px solid rgba(96, 165, 250, 0.5);
}

/* Average Rating - Orange/Gold */
div[data-testid="stMetric"]:nth-of-type(2) {
    background: linear-gradient(135deg, #f59e0b, #f97316);
    border: 1px solid rgba(251, 191, 36, 0.5);
}

/* Positive - Green */
div[data-testid="stMetric"]:nth-of-type(3) {
    background: linear-gradient(135deg, #059669, #22c55e);
    border: 1px solid rgba(74, 222, 128, 0.5);
}

/* Top Aspect - Purple */
div[data-testid="stMetric"]:nth-of-type(4) {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    border: 1px solid rgba(192, 132, 252, 0.5);
}

/* Card text */
div[data-testid="stMetric"] {
    padding: 20px 22px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

/* Hover effect */
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {
    color: white !important;
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
<div style="
    background: linear-gradient(
    135deg,
    #7c3aed 0%,
    #4f46e5 45%,
    #2563eb 100%
);
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 30px;
    text-align: center;
   box-shadow: 0 12px 35px rgba(79, 70, 229, 0.35);
">
<h1 style="color:white; margin:0; font-size:42px;">
🍽️ Restaurant Review Analyzer
</h1>

<p style="color:white; font-size:19px; margin-top:14px;">
AI Powered Restaurant Review Analysis & Rating Prediction
</p>

<p style="color:#e2e8f0; font-size:15px;">
 Analyzing <b>{len(df)}</b> cleaned reviews across
<b>{df["Business_Name"].nunique()}</b> restaurants
</p>

<div style="margin-top:20px;">
<span style="background:rgba(255,255,255,0.18); padding:10px 18px; border-radius:12px; margin:6px; color:white;">
 AI Powered
</span>

<span style="background:rgba(255,255,255,0.18); padding:10px 18px; border-radius:12px; margin:6px; color:white;">
 Sentiment Analysis
</span>

<span style="background:rgba(255,255,255,0.18); padding:10px 18px; border-radius:12px; margin:6px; color:white;">
 Dashboard Analytics
</span>
</div>

</div>
""",
    unsafe_allow_html=True
)
# =========================
# SIDEBAR FILTERS
# =========================

with st.sidebar:
    st.markdown("##  Filters")

    selected_city = st.selectbox(
        " City",
        ["Bangalore"]
    )

    restaurant_options = ["All"] + sorted(
        df["Business_Name"].dropna().unique().tolist()
    )

    selected_restaurant = st.selectbox(
        " Restaurant",
        restaurant_options
    )

    aspect_options = ["Ambience", "Food", "General", "Price", "Service"]

    selected_aspects = st.multiselect(
        " Aspects",
        aspect_options,
        default=aspect_options
    )
    st.markdown(
    """
<div class="sidebar-footer">
    <div> Data Analytics • AI • NLP</div>
    <div class="project-name">Restaurant Review Analyzer</div>
</div>
""",
    unsafe_allow_html=True
)

# =========================
# FILTERED DATASET
# =========================

subset = df.copy()

if selected_restaurant != "All":
    subset = subset[
        subset["Business_Name"] == selected_restaurant
    ]

if selected_aspects:
    subset = subset[
        subset["aspect"].isin(selected_aspects)
    ]
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

c1.metric(" Reviews", review_count)
c2.metric(" Avg. Rating", avg_rating)
c3.metric(" Positive", positive_pct)
c4.metric(" Top Aspect", top_aspect_value)
a, b = st.columns(2)
with a:
    st.subheader(" Rating Distribution")

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
                color=alt.Color(
                  "Rating:N",
                  scale=alt.Scale(
                  domain=[1, 2, 3, 4, 5],
                  range=["#ff4b5c", "#ff8c42", "#ffd93d", "#4dd599", "#00d4ff"]
                     ),
                     legend=None
                ),
                tooltip=[
                    alt.Tooltip("Rating:O", title="Rating"),
                    alt.Tooltip("Reviews:Q", title="Reviews"),
                ],
            )
            .properties(height=320)
        )

        st.altair_chart(rating_chart, use_container_width=True)
with b:
    st.subheader(" Reviews by Aspect")

    if len(subset):
        aspect_data = (
        subset["aspect"]
        .value_counts()
        .rename_axis("Aspect")
        .reset_index(name="Reviews")
    )

    aspect_chart = (
        alt.Chart(aspect_data)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6
        )
        .encode(
            x=alt.X("Aspect:N", title="Aspect", sort="-y"),
            y=alt.Y("Reviews:Q", title="Number of Reviews"),

            color=alt.Color(
                "Aspect:N",
                scale=alt.Scale(
                    domain=["Food", "Service", "General", "Price", "Ambience"],
                    range=["#ff5f6d", "#00d4ff", "#a855f7", "#ffd93d", "#4dd599"]
                ),
                legend=None
            ),

            tooltip=[
                alt.Tooltip("Aspect:N", title="Aspect"),
                alt.Tooltip("Reviews:Q", title="Reviews")
            ]
        )
        .properties(height=320)
    )

    st.altair_chart(aspect_chart, use_container_width=True)
st.subheader(" Average Rating by Restaurant")

rating_by_restaurant = (
    subset.groupby("Business_Name")["Rating"]
    .mean()
    .round(2)
    .reset_index()
)

restaurant_chart = (
    alt.Chart(rating_by_restaurant)
    .mark_bar(
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6
    )
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

        color=alt.Color(
            "Rating:Q",
            scale=alt.Scale(
                domain=[1, 2, 3, 4, 5],
                range=[
                    "#ff4b5c",
                    "#ff8c42",
                    "#ffd93d",
                    "#4dd599",
                    "#00d4ff"
                ]
            ),
            legend=None
        ),

        tooltip=[
            alt.Tooltip("Business_Name:N", title="Restaurant"),
            alt.Tooltip("Rating:Q", title="Average Rating", format=".2f")
        ]
    )
    .properties(height=400)
)

st.altair_chart(restaurant_chart, use_container_width=True)
st.subheader(" Word Cloud")

text = " ".join(subset["Review text"].astype(str))

if text.strip():
    wc = WordCloud(
        width=1200,
        height=450,
        background_color="#0E1117",
        colormap="turbo",
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
st.markdown(
    """
<div style="background: linear-gradient(135deg, #4c1d95, #7e22ce, #be185d); padding: 26px 30px; border-radius: 18px; margin-top: 28px; margin-bottom: 22px; box-shadow: 0 10px 30px rgba(109,40,217,0.25); border: 1px solid rgba(255,255,255,0.15);">
<h2 style="color:white; margin:0 0 8px 0; font-size:32px;"> Analyze & Predict</h2>
<p style="color:#e9d5ff; margin:0; font-size:16px;">Enter a restaurant review and let AI analyze its sentiment, rating and key aspect.</p>
</div>
""",
    unsafe_allow_html=True
)
user_text = st.text_area(
    " Enter your review",
    placeholder="e.g. The biryani was amazing but the service was slow",
    height=130
)

if st.button(" Analyze Review", type="primary", use_container_width=True) and user_text:

    if len(user_text.split()) < 6:
        st.info("Short review - prediction is based on very little text.")

    cleaned = clean_review_text(user_text)
    proba = pipeline.predict_proba([cleaned])[0]

    rating, sentiment, sent_probs = probs_to_outputs(proba, CLASSES)
    aspect, all_aspects, _ = tag_aspects(user_text)
    oov = oov_ratio(user_text, word_vec)

    st.markdown(
        """
<div style="margin-top:28px; margin-bottom:18px; padding:14px 20px;
border-left:5px solid #a855f7;
background:rgba(168,85,247,0.10);
border-radius:10px;">
<h3 style="margin:0; color:white;"> Analysis Results</h3>
<p style="margin:5px 0 0 0; color:#d8b4fe;">
AI-powered insights from your restaurant review
</p>
</div>
""",
        unsafe_allow_html=True
    )

    k1, k2, k3 = st.columns(3)

    k1.metric(" Predicted Sentiment", sentiment)
    k2.metric(" Predicted Rating", f"{rating:.1f} / 5")
    k3.metric(" Predicted Aspect", aspect)
        
    

    st.markdown(
            "** Star distribution:** " +
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