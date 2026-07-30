# ============================================================
#  LOCAL BUSINESS REVIEW ANALYZER - PROFESSIONAL DASHBOARD
#  Run:  streamlit run app/streamlit_app.py
# ============================================================
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

from text_utils import (CLEAN_CSV, MODELS_DIR, clean_review_text, oov_ratio,
                        probs_to_outputs, rating_to_sentiment, tag_aspects,
                        LexiconFeatures, POS_WORDS, NEG_WORDS, NEGATORS)

try:
    import gensim
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

st.set_page_config(page_title="Restaurant Review Analyzer", page_icon="🍽️", layout="wide")

# ============================================================
#  THEME
# ============================================================
SENT_COLORS = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}
RATING_COLORS = {1: "#ef4444", 2: "#f97316", 3: "#f59e0b", 4: "#84cc16", 5: "#22c55e"}
TOPIC_COLORS = ["#6366f1", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9"]
DEFAULT_CITIES = ["Bangalore"]   # ← more cities auto-appear when a City column exists

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: #f6f7fb; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a,#1e293b); }
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stRadio label span { color:#e2e8f0 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color:#fff !important; }

.hero { background: linear-gradient(135deg,#1e1b4b 0%,#312e81 45%,#0f766e 100%);
  border-radius:20px; padding:2.2rem 2.5rem; color:#fff;
  box-shadow:0 12px 32px rgba(30,27,75,.28); margin-bottom:1.5rem; }
.hero h1 { font-size:2.5rem; font-weight:800; margin:0; letter-spacing:-.02em; }
.hero p { opacity:.88; margin-top:.5rem; font-size:1.05rem; }
.hero-badges span { display:inline-block; background:rgba(255,255,255,.12);
  border:1px solid rgba(255,255,255,.22); padding:.38rem .9rem; border-radius:999px;
  margin:.7rem .45rem 0 0; font-size:.85rem; font-weight:500; }

.kpi { background:#fff; border-radius:14px; padding:1.1rem 1.25rem;
  box-shadow:0 2px 12px rgba(15,23,42,.07); height:100%; border-left:5px solid #6366f1; }
.kpi-top { display:flex; align-items:center; gap:.45rem; }
.kpi-icon { font-size:1.25rem; }
.kpi-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; color:#64748b; font-weight:700; }
.kpi-value { font-size:1.9rem; font-weight:800; color:#0f172a; margin-top:.25rem; line-height:1.1; }
.kpi-sub { font-size:.8rem; color:#94a3b8; margin-top:.2rem; }

.section-title { font-size:1.3rem; font-weight:700; color:#0f172a; margin:1.6rem 0 .7rem; }
.card { background:#fff; border-radius:16px; padding:1.1rem 1.3rem;
  box-shadow:0 2px 12px rgba(15,23,42,.06); margin-bottom:1rem; }
.card h3 { margin:0 0 .3rem; font-size:1.05rem; color:#0f172a; }
.card p.cap { margin:0; font-size:.82rem; color:#94a3b8; }

.chip { display:inline-block; background:#eef2ff; color:#4338ca; border-radius:999px;
  padding:.32rem .85rem; margin:.18rem; font-size:.8rem; font-weight:600; }
.stButton>button { background:linear-gradient(90deg,#6366f1,#14b8a6) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  padding:.65rem 1rem !important; font-weight:700 !important; }
.stButton>button:hover { opacity:.92; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  DATA + MODEL LOADING  (same backend as your original code)
# ============================================================
@st.cache_data
def load_data():
    d = pd.read_csv(CLEAN_CSV)
    d["Business_Name"] = d["Business_Name"].fillna("Unlisted").astype(str).str.strip()
    d["sentiment_label"] = d["Rating"].apply(rating_to_sentiment)
    if "aspect" not in d.columns:
        d["aspect"] = d["Review text"].astype(str).apply(lambda t: tag_aspects(t)[0])
    # City support: uses a City column if present, else default (future-ready)
    city_col = next((c for c in d.columns if c.lower() == "city"), None)
    d["city"] = d[city_col].fillna("Bangalore").astype(str) if city_col else "Bangalore"
    d["Sentiment"] = d["sentiment_label"].astype(str).str.title()
    return d

@st.cache_resource
def load_model():
    b = joblib.load(os.path.join(MODELS_DIR, "review_model.pkl"))
    return b["pipeline"], np.asarray(b["classes"])

@st.cache_resource
def load_lda():
    if not HAS_GENSIM:
        return None
    lp = os.path.join(MODELS_DIR, "lda_model.gensim")
    dp = os.path.join(MODELS_DIR, "lda_dictionary.gensim")
    tp = os.path.join(MODELS_DIR, "topic_labels.json")
    if not (os.path.exists(lp) and os.path.exists(dp)):
        return None
    try:
        out = {"lda": gensim.models.LdaModel.load(lp),
               "dict": gensim.corpora.Dictionary.load(dp)}
        if os.path.exists(tp):
            with open(tp, encoding="utf-8") as f:
                out["labels"] = json.load(f)
        else:
            out["labels"] = {}
        return out
    except Exception:
        return None

def style(fig, height=400):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=35, b=10),
                      font=dict(family="Inter", size=13, color="#334155"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.18))
    return fig

def kpi(icon, label, value, sub="", color="#6366f1"):
    return (f'<div class="kpi" style="border-left-color:{color};">'
            f'<div class="kpi-top"><span class="kpi-icon">{icon}</span>'
            f'<span class="kpi-label">{label}</span></div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')

def top_words(series, n=10):
    words = " ".join(series.astype(str)).lower().split()
    words = [w for w in words if len(w) > 3 and w not in STOPWORDS]
    return pd.DataFrame(Counter(words).most_common(n), columns=["word", "count"])

df = load_data()
pipeline, CLASSES = load_model()
word_vec = pipeline.named_steps["feats"].transformer_list[0][1]
lda = load_lda()

# ============================================================
#  SIDEBAR : NAVIGATION + FILTERS
# ============================================================
with st.sidebar:
    st.markdown("## 🍽️ Review Analyzer")
    st.caption("Local Business Intelligence Platform")
    page = st.radio("Navigate", [
        "🏠 Overview", "💬 Sentiment Analysis", "⭐ Ratings & Restaurants",
        "☁️ Topics & Keywords", "🔍 Review Analyzer", "ℹ️ About & Methodology"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🎛️ Filters")

    city_options = sorted(df["city"].dropna().unique().tolist()) or DEFAULT_CITIES
    city = st.selectbox("📍 City", ["All Cities"] + city_options,
                        help="Add a 'City' column to the dataset to enable more cities.")

    df_city = df if city == "All Cities" else df[df["city"] == city]
    rest_opts = sorted(df_city["Business_Name"].unique().tolist())
    restaurant = st.selectbox("🏪 Restaurant", ["All"] + rest_opts)

    aspect_opts = sorted(df_city["aspect"].dropna().unique().tolist())
    aspects = st.multiselect("🧩 Aspects", aspect_opts, default=aspect_opts)

    st.markdown("---")
    st.caption("Data Analytics • AI • NLP\nKanika • Priyal • Faizan • Rishabh")

subset = df_city.copy()
if restaurant != "All":
    subset = subset[subset["Business_Name"] == restaurant]
if aspects:
    subset = subset[subset["aspect"].isin(aspects)]

# ============================================================
#  PAGE 1 : OVERVIEW
# ============================================================
if page == "🏠 Overview":
    pos_pct = (subset["sentiment_label"] == "positive").mean() * 100 if len(subset) else 0
    top_aspect = subset["aspect"].mode().iloc[0] if len(subset) else "—"

    st.markdown(f"""
    <div class="hero"><h1>🍽️ Restaurant Review Analyzer</h1>
    <p>AI-powered customer intelligence — sentiment, ratings & topic insights from real reviews.</p>
    <div class="hero-badges"><span>📍 {city}</span>
    <span>🏪 {subset['Business_Name'].nunique()} restaurants</span>
    <span>📝 {len(subset)} reviews</span><span>🤖 NLP + Machine Learning</span></div></div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("📝", "Total Reviews", f"{len(subset):,}", "cleaned & analyzed", "#6366f1"), unsafe_allow_html=True)
    c2.markdown(kpi("⭐", "Avg Rating", f"{subset['Rating'].mean():.2f}", "out of 5.0", "#f59e0b"), unsafe_allow_html=True)
    c3.markdown(kpi("😊", "Positive Share", f"{pos_pct:.1f}%", "of all reviews", "#22c55e"), unsafe_allow_html=True)
    c4.markdown(kpi("🏪", "Restaurants", subset["Business_Name"].nunique(), f"in {city}", "#0ea5e9"), unsafe_allow_html=True)
    c5.markdown(kpi("🧩", "Top Aspect", top_aspect, "most discussed", "#8b5cf6"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Rating & Sentiment Snapshot</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="card"><h3>Rating Distribution</h3><p class="cap">How customers rated their experience (1–5 stars)</p></div>', unsafe_allow_html=True)
        if len(subset):
            rc = subset["Rating"].astype(int).value_counts().sort_index()
            fig = px.bar(x=rc.index, y=rc.values, color=rc.index,
                         color_discrete_map=RATING_COLORS,
                         labels={"x": "Rating", "y": "Reviews"}, text=rc.values)
            fig.update_traces(marker=dict(cornerradius=8))
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 360), use_container_width=True)
    with b:
        st.markdown('<div class="card"><h3>Sentiment Split</h3><p class="cap">Positive vs Neutral vs Negative</p></div>', unsafe_allow_html=True)
        if len(subset):
            sc = subset["Sentiment"].value_counts()
            fig = go.Figure(go.Pie(labels=sc.index, values=sc.values, hole=0.62,
                                   marker=dict(colors=[SENT_COLORS.get(s, "#94a3b8") for s in sc.index])))
            fig.update_traces(textinfo="label+percent", hoverinfo="label+value")
            fig.add_annotation(text=f"{len(subset)}", x=0.5, y=0.5, font_size=26,
                               showarrow=False, font_family="Inter")
            st.plotly_chart(style(fig, 360), use_container_width=True)

    st.markdown('<div class="section-title">🧩 Aspect Intelligence</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="card"><h3>Reviews by Aspect</h3><p class="cap">What customers talk about most</p></div>', unsafe_allow_html=True)
        if len(subset):
            ac = subset["aspect"].value_counts().reset_index()
            ac.columns = ["aspect", "count"]
            fig = px.bar(ac, x="count", y="aspect", orientation="h", color="count",
                         color_continuous_scale="Teal", text="count")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(style(fig, 360), use_container_width=True)
    with b:
        st.markdown('<div class="card"><h3>Sentiment by Aspect</h3><p class="cap">Where the complaints hide</p></div>', unsafe_allow_html=True)
        if len(subset):
            sa = subset.groupby(["aspect", "Sentiment"]).size().reset_index(name="count")
            fig = px.bar(sa, x="aspect", y="count", color="Sentiment",
                         color_discrete_map=SENT_COLORS, barmode="stack")
            st.plotly_chart(style(fig, 360), use_container_width=True)

    st.markdown('<div class="section-title">🏆 Restaurant Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Average Rating by Restaurant</h3><p class="cap">Green = higher rated · Red line = overall average</p></div>', unsafe_allow_html=True)
    if len(df_city) and df_city["Business_Name"].nunique() > 1:
        ar = df_city.groupby("Business_Name")["Rating"].mean().sort_values().reset_index()
        fig = px.bar(ar, x="Rating", y="Business_Name", orientation="h", color="Rating",
                     color_continuous_scale="RdYlGn", range_color=[1, 5],
                     text=ar["Rating"].round(2))
        fig.add_vline(x=df_city["Rating"].mean(), line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Avg {df_city['Rating'].mean():.2f}")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig, 420), use_container_width=True)

        best, worst = ar.iloc[-1], ar.iloc[0]
        neg = subset[subset["Sentiment"] == "Negative"]
        pain = neg["aspect"].mode().iloc[0] if len(neg) else "—"
        st.markdown('<div class="section-title">💡 Auto-Generated Insights</div>', unsafe_allow_html=True)
        st.success(f"🏆 **{best['Business_Name']}** leads with {best['Rating']:.2f}★ average rating.")
        st.warning(f"⚠️ **{pain}** is the most complained-about aspect in negative reviews.")
        st.info(f"📈 {pos_pct:.1f}% of customers left positive feedback — "
                f"{'strong' if pos_pct > 70 else 'moderate'} satisfaction level.")

# ============================================================
#  PAGE 2 : SENTIMENT ANALYSIS
# ============================================================
elif page == "💬 Sentiment Analysis":
    st.markdown('<div class="section-title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
    a, b = st.columns([1, 2])
    with a:
        if len(subset):
            sc = subset["Sentiment"].value_counts()
            fig = go.Figure(go.Pie(labels=sc.index, values=sc.values, hole=0.6,
                                   marker=dict(colors=[SENT_COLORS.get(s, "#94a3b8") for s in sc.index])))
            fig.update_traces(textinfo="label+percent+value")
            st.plotly_chart(style(fig, 380), use_container_width=True)
    with b:
        if len(subset):
            sa = subset.groupby(["Business_Name", "Sentiment"]).size().reset_index(name="count")
            fig = px.bar(sa, x="Business_Name", y="count", color="Sentiment",
                         color_discrete_map=SENT_COLORS, barmode="group")
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(style(fig, 380), use_container_width=True)

    st.markdown('<div class="card"><h3>Aspect → Sentiment Sunburst</h3><p class="cap">Click a slice to drill down</p></div>', unsafe_allow_html=True)
    if len(subset):
        fig = px.sunburst(subset, path=["aspect", "Sentiment"], color="Sentiment",
                          color_discrete_map=SENT_COLORS)
        st.plotly_chart(style(fig, 460), use_container_width=True)

    st.markdown('<div class="section-title">🔤 Most Frequent Words by Sentiment</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="card"><h3 style="color:#16a34a;">😊 Positive Reviews</h3></div>', unsafe_allow_html=True)
        pw = top_words(subset[subset["Sentiment"] == "Positive"]["review_text"
                          if "review_text" in subset.columns else "Review text"])
        if len(pw):
            fig = px.bar(pw, x="count", y="word", orientation="h", color="count",
                         color_continuous_scale="Greens", text="count")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(style(fig, 380), use_container_width=True)
    with p2:
        st.markdown('<div class="card"><h3 style="color:#dc2626;">😞 Negative Reviews</h3></div>', unsafe_allow_html=True)
        nw = top_words(subset[subset["Sentiment"] == "Negative"]["review_text"
                          if "review_text" in subset.columns else "Review text"])
        if len(nw):
            fig = px.bar(nw, x="count", y="word", orientation="h", color="count",
                         color_continuous_scale="Reds", text="count")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(style(fig, 380), use_container_width=True)

# ============================================================
#  PAGE 3 : RATINGS & RESTAURANTS
# ============================================================
elif page == "⭐ Ratings & Restaurants":
    st.markdown('<div class="section-title">⭐ Ratings & Restaurant Performance</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="card"><h3>Overall Rating Gauge</h3></div>', unsafe_allow_html=True)
        if len(subset):
            fig = go.Figure(go.Indicator(mode="gauge+number",
                                         value=round(subset["Rating"].mean(), 2),
                                         number={"font": {"size": 40}},
                                         gauge=dict(axis=dict(range=[0, 5]), bar=dict(color="#6366f1"),
                                                    steps=[{"range": [0, 2], "color": "#fee2e2"},
                                                           {"range": [2, 3.5], "color": "#fef9c3"},
                                                           {"range": [3.5, 5], "color": "#dcfce7"}])))
            st.plotly_chart(style(fig, 320), use_container_width=True)
    with g2:
        st.markdown('<div class="card"><h3>Rating vs Sentiment</h3><p class="cap">Do ratings match written sentiment?</p></div>', unsafe_allow_html=True)
        if len(subset):
            fig = px.box(subset, x="Sentiment", y="Rating", color="Sentiment",
                         color_discrete_map=SENT_COLORS, points="outliers")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 320), use_container_width=True)

    st.markdown('<div class="card"><h3>Restaurant Profile Radar</h3><p class="cap">Average rating per aspect — selected restaurant vs overall</p></div>', unsafe_allow_html=True)
    if len(subset):
        overall = subset.groupby("aspect")["Rating"].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=overall.values, theta=overall.index, fill="toself",
                                      name="Overall", line=dict(color="#94a3b8", dash="dash")))
        if restaurant != "All":
            sel = subset.groupby("aspect")["Rating"].mean()
            fig.add_trace(go.Scatterpolar(r=sel.values, theta=sel.index, fill="toself",
                                          name=restaurant, line=dict(color="#6366f1")))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))
        st.plotly_chart(style(fig, 480), use_container_width=True)

    st.markdown('<div class="card"><h3>Restaurant Comparison Table</h3></div>', unsafe_allow_html=True)
    if len(df_city):
        tbl = df_city.groupby("Business_Name").agg(
            Reviews=("Rating", "size"), Avg_Rating=("Rating", "mean"),
            Positive_Pct=("sentiment_label", lambda s: round((s == "positive").mean() * 100, 1))
        ).sort_values("Avg_Rating", ascending=False).reset_index()
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ============================================================
#  PAGE 4 : TOPICS & KEYWORDS
# ============================================================
elif page == "☁️ Topics & Keywords":
    st.markdown('<div class="section-title">☁️ Topic Modeling & Keywords</div>', unsafe_allow_html=True)
    review_col = "Review text" if "Review text" in subset.columns else "review_text"

    st.markdown('<div class="card"><h3>Review Word Cloud</h3><p class="cap">Bigger word = more frequent in reviews</p></div>', unsafe_allow_html=True)
    text = " ".join(subset[review_col].astype(str))
    if text.strip():
        wc = WordCloud(width=1400, height=520, background_color="#0f172a",
                       colormap="cool", stopwords=set(STOPWORDS),
                       max_words=160, collocations=False).generate(text)
        fig, ax = plt.subplots(figsize=(14, 5.2))
        fig.patch.set_facecolor("#0f172a")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if lda:
        st.markdown('<div class="card"><h3>LDA Topic Distribution</h3><p class="cap">Share of each hidden topic across the corpus</p></div>', unsafe_allow_html=True)
        corpus = [lda["dict"].doc2bow(str(r).lower().split())
                  for r in subset[review_col].head(300)]
        weights = Counter()
        for bow in corpus:
            for tid, w in lda["lda"][bow]:
                weights[tid] += w
        names = [str(lda["labels"].get(str(t), lda["labels"].get(t, f"Topic {t}"))) for t in weights]
        fig = px.pie(names=names, values=list(weights.values()), hole=0.5,
                     color_discrete_sequence=TOPIC_COLORS)
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(style(fig, 420), use_container_width=True)

        st.markdown('<div class="card"><h3>Top Keywords per Topic</h3></div>', unsafe_allow_html=True)
        rows = []
        for tid, _ in lda["lda"].print_topics(-1, 6):
            kws = [w for w, _ in lda["lda"].show_topic(tid, 6)]
            lbl = lda["labels"].get(str(tid), lda["labels"].get(tid, f"Topic {tid}"))
            rows.append({"Topic": lbl, "Keywords": ", ".join(kws)})
        st.table(pd.DataFrame(rows))
    else:
        st.info("LDA model not loaded — ensure lda_model.gensim & lda_dictionary.gensim exist in models/.")

# ============================================================
#  PAGE 5 : REVIEW ANALYZER  (your original logic, upgraded UI)
# ============================================================
elif page == "🔍 Review Analyzer":
    st.markdown('<div class="section-title">🔍 Analyze Your Own Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><p class="cap">Type any restaurant review and the AI pipeline predicts its sentiment, star rating and dominant topic in real time.</p></div>', unsafe_allow_html=True)

    user_text = st.text_area("✍️ Enter your review", height=140,
                             placeholder="e.g. The biryani was amazing but the service was slow")

    if st.button("✨ Analyze Review", type="primary", use_container_width=True) and user_text.strip():

        if len(user_text.split()) < 6:
            st.info("Short review - prediction is based on very little text.")

        cleaned = clean_review_text(user_text)
        proba = pipeline.predict_proba([cleaned])[0]
        rating, sentiment, sent_probs = probs_to_outputs(proba, CLASSES)
        aspect, all_aspects, _ = tag_aspects(user_text)
        oov = oov_ratio(user_text, word_vec)

        sent_title = str(sentiment).title()
        emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}.get(sent_title, "😐")
        stars = "★" * int(round(rating)) + "☆" * (5 - int(round(rating)))

        k1, k2, k3 = st.columns(3)
        k1.markdown(kpi(emoji, "Predicted Sentiment", sent_title,
                        f"confidence {max(sent_probs.values()):.0%}",
                        SENT_COLORS.get(sent_title, "#6366f1")), unsafe_allow_html=True)
        k2.markdown(kpi("⭐", "Predicted Rating", stars, f"{rating:.1f} / 5.0", "#f59e0b"), unsafe_allow_html=True)
        k3.markdown(kpi("🧩", "Dominant Topic", aspect, "via aspect tagging / LDA", "#8b5cf6"), unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(mode="gauge+number", value=rating,
                                     number={"suffix": " / 5"},
                                     gauge=dict(axis=dict(range=[0, 5]),
                                                bar=dict(color=SENT_COLORS.get(sent_title, "#6366f1")),
                                                steps=[{"range": [0, 2], "color": "#fee2e2"},
                                                       {"range": [2, 3.5], "color": "#fef9c3"},
                                                       {"range": [3.5, 5], "color": "#dcfce7"}])))
        st.plotly_chart(style(fig, 280), use_container_width=True)

        st.markdown('<div class="card"><h3>🔎 Prediction Details</h3></div>', unsafe_allow_html=True)
        st.markdown("**Sentiment confidence:** " +
                    " · ".join(f"{k.title()} **{v:.0%}**" for k, v in sent_probs.items()))
        st.markdown("**⭐ Star distribution:** " +
                    " · ".join(f"{int(c)}★ **{p:.0%}**" for c, p in zip(CLASSES, proba)))
        if all_aspects:
            st.caption("Aspects mentioned: " + ", ".join(all_aspects))

        ordered = sorted(sent_probs.values(), reverse=True)
        if oov > 0.6:
            st.warning(f"{oov:.0%} of the words here were not seen during training — "
                       "the prediction relies mostly on general polarity cues.")
        if len(ordered) > 1 and ordered[0] - ordered[1] < 0.15:
            st.warning("Sentiment classes were close — treat this as uncertain.")

        toks = set(cleaned.split())
        has_pos = bool(toks & POS_WORDS)
        has_neg = bool(toks & NEG_WORDS) or bool(toks & NEGATORS)
        if len(all_aspects) > 1 and has_pos and has_neg:
            st.info(f"Mixed review — mentions {' and '.join(all_aspects)} with both "
                    "positive and negative points. The single score averages them.")
        elif len(all_aspects) > 1:
            st.info(f"Covers several aspects: {' and '.join(all_aspects)}.")

# ============================================================
#  PAGE 6 : ABOUT & METHODOLOGY
# ============================================================
elif page == "ℹ️ About & Methodology":
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card"><h3>🎯 Project Summary</h3>
<p>The <b>Local Business Review Analyzer</b> converts manually collected, unstructured restaurant reviews
into actionable business intelligence using Data Analytics, Machine Learning and NLP. It performs sentiment
classification, star-rating prediction and LDA topic modeling, and presents the results through this
interactive dashboard.</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>🔄 End-to-End Pipeline</h3></div>', unsafe_allow_html=True)
    st.code("Manual Collection → Merge CSVs → Clean & Preprocess (NLP) → TF-IDF + Lexicon Features → "
            "Sentiment Classification → Rating Prediction → LDA Topic Modeling → Streamlit Dashboard")

    st.markdown('<div class="card"><h3>🛠️ Technology Stack</h3></div>', unsafe_allow_html=True)
    tech = ["Python", "Pandas", "NumPy", "Scikit-learn", "NLTK", "Gensim", "Plotly",
            "WordCloud", "Streamlit", "Jupyter", "Git & GitHub"]
    st.markdown("".join(f'<span class="chip">{t}</span>' for t in tech), unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>🤖 Models Used</h3></div>', unsafe_allow_html=True)
    st.markdown("""
- **Sentiment Classification:** Logistic Regression · Multinomial Naive Bayes
- **Rating Prediction:** Linear Regression · Random Forest · XGBoost
- **Topic Modeling:** Latent Dirichlet Allocation (LDA)
- **Feature Extraction:** TF-IDF (unigrams + bigrams) + Lexicon features""")

    st.markdown('<div class="card"><h3>👥 Team</h3></div>', unsafe_allow_html=True)
    st.table(pd.DataFrame({"Member": ["Kanika", "Priyal", "Faizan", "Rishabh"],
                           "Role": ["Data Collection & Analysis"] * 4}))

    st.markdown('<div class="card"><h3>🚀 Future Scope</h3></div>', unsafe_allow_html=True)
    st.markdown("""
- 📍 Multi-city expansion (city selector already built-in)
-  BERT / transformer-based sentiment models
- 🎯 Aspect-based sentiment analysis
- 🌐 Multilingual review support & cloud deployment""")

st.markdown("---")