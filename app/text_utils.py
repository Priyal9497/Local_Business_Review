# ============================================================
# text_utils.py
# SINGLE SOURCE OF TRUTH for text preprocessing across all three
# pipelines: sentiment classification, rating prediction, and
# topic modeling.
# ============================================================

import re

# ------------------------------------------------------------
# Sentiment / rating preprocessing (negation-aware)
# ------------------------------------------------------------
NEGATION_WORDS = (
    r"not|no|never|without|don't|doesn't|didn't|isn't|aren't|wasn't|"
    r"weren't|won't|wouldn't|couldn't|can't|cannot"
)

_NEGATION_PATTERN = re.compile(
    rf"\b({NEGATION_WORDS})\s+(\w+)", flags=re.IGNORECASE
)


def fix_negations(text: str) -> str:
    """
    Merge a negation word with the word that follows it into a single
    underscore-joined token, e.g. "not good" -> "not_good".
    Must be applied identically in the sentiment notebook, the rating
    notebook, and app.py.
    """
    text = str(text).lower()
    return _NEGATION_PATTERN.sub(r"\1_\2", text)


def clean_review_text(text: str) -> str:
    """Full cleaning pipeline for the sentiment/rating models."""
    text = str(text).strip()
    text = fix_negations(text)
    text = re.sub(r"\s+", " ", text)
    return text


def oov_ratio(text: str, vectorizer) -> float:
    """
    Diagnostic: fraction of tokens in `text` that are NOT in the
    vectorizer's vocabulary. High ratio = prediction is based on very
    little real signal.
    """
    analyzer = vectorizer.build_analyzer()
    tokens = analyzer(text)
    if not tokens:
        return 1.0
    vocab = vectorizer.vocabulary_
    unknown = sum(1 for t in tokens if t not in vocab)
    return unknown / len(tokens)


# ------------------------------------------------------------
# Topic modeling preprocessing
# Must be IDENTICAL to the tokenize() function in the corrected
# topic modeling notebook, or the app's live topic predictions
# won't match what the model was actually trained on.
# ------------------------------------------------------------
try:
    from nltk.corpus import stopwords
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _STOP_WORDS = set(stopwords.words("english"))

_WORD_RE = re.compile(r"\b[a-z]+\b")


def tokenize_for_topics(text: str) -> list:
    """Same tokenizer used to build the LDA dictionary/corpus."""
    text = str(text).lower()
    words = _WORD_RE.findall(text)
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


def get_dominant_topic(text: str, lda_model, dictionary):
    """
    Run the same tokenize -> bag-of-words -> LDA inference pipeline
    used in the notebook, for a single piece of live text.
    Returns (topic_id, probability) or (None, None) if no topic
    could be inferred (e.g. text too short after cleaning).
    """
    tokens = tokenize_for_topics(text)
    bow = dictionary.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow)
    if not topics:
        return None, None
    topic_id, prob = max(topics, key=lambda x: x[1])
    return topic_id, prob