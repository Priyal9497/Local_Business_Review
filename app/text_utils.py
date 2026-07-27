# ============================================================
# app/text_utils.py — single source of truth for ALL pipelines
# ============================================================
import os
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def data_path(*parts):
    for folder in ("Data", "data"):
        p = os.path.join(PROJECT_ROOT, folder, *parts)
        if os.path.exists(p):
            return p
    return os.path.join(PROJECT_ROOT, "Data", *parts)


CLEAN_CSV = data_path("processed", "cleaned_restaurant_reviews.csv")
MERGED_CSV = data_path("processed", "merged_restaurant_reviews.csv")

# ------------------------------------------------------------
# Negation (matches BOTH "don't" and "dont" — your cleaning stripped apostrophes)
# ------------------------------------------------------------
NEGATORS = {
    "not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
    "without", "hardly", "barely", "scarcely", "lack", "lacks", "lacking",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "wont", "wouldnt", "couldnt", "cant", "cannot", "shouldnt", "havent",
    "hasnt", "hadnt", "aint",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "wouldn't", "couldn't", "can't", "shouldn't", "haven't",
}
CLAUSE_BREAKS = {"but", "however", "although", "though", "yet", "still",
                 "except", "unless", "while", "whereas"}
INTENSIFIERS = {"very", "really", "extremely", "so", "soo", "sooo", "too",
                "highly", "absolutely", "totally", "super", "quite", "much"}
NEG_WINDOW = 3
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text):
    return _TOKEN_RE.findall(str(text).lower())


def negation_markers(tokens, window=NEG_WINDOW):
    out, scope = [], 0
    for t in tokens:
        if t in NEGATORS:
            scope = window
            continue
        if t in CLAUSE_BREAKS:
            scope = 0
        if scope > 0:
            if t not in INTENSIFIERS:
                out.append("neg_" + t)
            scope -= 1
    return out


def clean_review_text(text):
    """Keeps original tokens AND appends neg_ markers (old code destroyed the word)."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)      # SPACE, not "" — no more "burgera"
    text = re.sub(r"\s+", " ", text).strip()
    toks = tokenize(text)
    markers = negation_markers(toks)
    return (text + " " + " ".join(markers)).strip() if markers else text


# ------------------------------------------------------------
# Sentiment lexicon — makes negation work on unseen words
# ------------------------------------------------------------
POS_WORDS = {
    "good", "great", "amazing", "awesome", "excellent", "delicious", "tasty",
    "yummy", "fantastic", "wonderful", "lovely", "perfect", "best", "nice",
    "fresh", "flavorful", "flavourful", "authentic", "recommend", "recommended",
    "loved", "love", "liked", "like", "enjoyed", "enjoy", "friendly", "polite",
    "quick", "fast", "clean", "cozy", "pleasant", "worth", "affordable",
    "reasonable", "generous", "hot", "crispy", "soft", "juicy", "must",
    "favourite", "favorite", "satisfying", "happy", "impressed", "superb",
    "outstanding", "brilliant", "top", "well", "helpful", "attentive",
}
NEG_WORDS = {
    "bad", "worst", "terrible", "awful", "horrible", "poor", "disgusting",
    "bland", "tasteless", "stale", "cold", "oily", "greasy", "burnt", "raw",
    "soggy", "rude", "slow", "late", "dirty", "unhygienic", "noisy", "crowded",
    "expensive", "overpriced", "costly", "pricey", "disappointing",
    "disappointed", "avoid", "waste", "pathetic", "mediocre", "average",
    "ordinary", "hyped", "overrated", "rubbish", "unfriendly", "careless",
    "unprofessional", "wrong", "missing", "small", "tiny", "hard", "dry",
    "spoiled", "sick", "problem", "issue", "complaint", "worse", "boring",
}


class LexiconFeatures(BaseEstimator, TransformerMixin):
    N_FEATURES = 7

    def fit(self, X, y=None):
        return self

    def _score_one(self, text):
        toks = [t for t in tokenize(text) if not t.startswith("neg_")]
        if not toks:
            return [0.0] * self.N_FEATURES
        pos = neg = 0.0
        scope = 0
        weight = 1.0
        n_neg_words = 0
        n_intens = 0
        for t in toks:
            if t in CLAUSE_BREAKS:
                weight = 1.5
                scope = 0
                continue
            if t in NEGATORS:
                scope = NEG_WINDOW
                n_neg_words += 1
                continue
            if t in INTENSIFIERS:
                n_intens += 1
                continue
            polarity = 1.0 if t in POS_WORDS else (-1.0 if t in NEG_WORDS else 0.0)
            if polarity != 0.0:
                if scope > 0:
                    polarity = -polarity * 0.8
                if polarity > 0:
                    pos += polarity * weight
                else:
                    neg += -polarity * weight
            if scope > 0:
                scope -= 1
        n = len(toks)
        net = pos - neg
        return [pos / n, neg / n, net / n, np.tanh(net),
                n_neg_words / n, n_intens / n, min(n, 60) / 60.0]

    def transform(self, X):
        return np.asarray([self._score_one(t) for t in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(["lex_pos", "lex_neg", "lex_net", "lex_tanh",
                         "lex_negators", "lex_intens", "lex_len"])


def oov_ratio(text, word_vectorizer):
    """Honest OOV: unigrams only (old version counted bigrams -> fake 60%)."""
    vocab = word_vectorizer.vocabulary_
    toks = [t for t in tokenize(text) if not t.startswith("neg_") and len(t) > 1]
    if not toks:
        return 1.0
    return sum(1 for t in toks if t not in vocab) / len(toks)


# ------------------------------------------------------------
# Aspect / topic tagging (replaces the broken LDA "Unknown" lookup)
# ------------------------------------------------------------
ASPECTS = {
    "Food": ["food", "taste", "tasty", "dish", "dishes", "biryani", "biriyani",
             "shawarma", "shwarma", "dosa", "idli", "vada", "tandoori", "chicken",
             "mutton", "paneer", "burger", "pizza", "cheese", "curry", "masala",
             "flavor", "flavour", "spicy", "delicious", "menu", "coffee", "chilli",
             "chutney", "sambar", "rice", "bread", "naan", "dessert", "sweet",
             "breakfast", "lunch", "dinner", "meal", "portion", "quantity", "fresh",
             "bland", "oily", "stale", "cooked", "snack", "starter", "thali"],
    "Service": ["service", "staff", "waiter", "waiters", "server", "served",
                "serving", "behaviour", "behavior", "rude", "polite", "friendly",
                "slow", "wait", "waiting", "attentive", "order", "ordered",
                "manager", "hospitality", "quick", "prompt", "delay", "delayed",
                "response", "helpful"],
    "Ambience": ["ambience", "ambiance", "atmosphere", "music", "decor", "interior",
                 "seating", "seats", "crowded", "clean", "cleanliness", "hygiene",
                 "vibe", "space", "spacious", "noisy", "lighting", "washroom",
                 "parking", "comfortable", "cozy", "environment"],
    "Price": ["price", "prices", "pricing", "expensive", "cheap", "costly", "cost",
              "value", "worth", "affordable", "budget", "overpriced", "pricey",
              "reasonable", "bill", "money", "rate", "rates"],
}


def tag_aspects(text):
    toks = set(tokenize(text))
    scores = {}
    for aspect, kws in ASPECTS.items():
        hits = len(toks & set(kws))
        if hits:
            scores[aspect] = hits
    if not scores:
        return "General", [], 0.0
    total = sum(scores.values())
    primary = max(scores, key=scores.get)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return primary, ordered, scores[primary] / total


def rating_to_sentiment(r):
    if r >= 3.5:
        return "positive"
    if r >= 2.5:
        return "neutral"
    return "negative"


def probs_to_outputs(proba, classes):
    """One probability vector -> rating + sentiment. They can never disagree now."""
    classes = np.asarray(classes, dtype=float)
    proba = np.asarray(proba, dtype=float)
    rating = float(np.clip((proba * classes).sum(), 1.0, 5.0))
    neg = float(proba[classes <= 2].sum())
    neu = float(proba[(classes > 2) & (classes < 4)].sum())
    pos = float(proba[classes >= 4].sum())
    sent_probs = {"negative": neg, "neutral": neu, "positive": pos}
    sentiment = max(sent_probs, key=sent_probs.get)
    return rating, sentiment, sent_probs