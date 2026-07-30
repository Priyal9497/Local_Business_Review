# Restaurant Review Sentiment & Aspect Analyzer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)

Data Science / NLP project that analyzes manually collected restaurant reviews to predict star ratings, derive sentiment, tag aspects (Food/Service/Ambience/Price), and surface latent topics — delivered via an interactive Streamlit dashboard.

## Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Run](#installation--run)
- [Dashboard Features](#dashboard-features)
- [Model Evaluation](#model-evaluation)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Ethics](#ethics)
- [License](#license)
- [Contact](#contact)

## Problem Statement

Star ratings alone don't explain *why* a customer was satisfied or not. This project converts raw review text into a predicted rating, sentiment label, and driving aspect, giving business owners actionable insight beyond a raw average rating.

## Dataset

Manually collected, rating-stratified reviews from **10 restaurants in Bengaluru** — no scraping, no personal identifiers (see `scraper/data_collection_notes.md`).

| Field | Description |
|---|---|
| `Business_Name` | Restaurant name |
| `Location` | Area/location |
| `Review text` | Raw review text |
| `Rating` | Star rating (1–5) |
| `review_clean` | Cleaned, negation-tagged text used for modeling |

**Cleaned dataset (n = 577):**

| Rating | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---:|---:|---:|---:|---:|
| Count | 57 | 36 | 52 | 144 | 288 |

## Pipeline

```text
Manual Review Collection
        │
        ▼
Cleaning & Validation (dedup, whitespace, length filter)
        │
        ▼
NLP Preprocessing (tokenize, stopwords, lemmatize, negation tagging)
        │
        ▼
TF-IDF Feature Union (word 1-gram/2-gram + char n-gram) + Sentiment Lexicon
        │
        ├──► Logistic Regression (rating prediction, tuned via GridSearchCV)
        │        └──► Sentiment derived from predicted rating
        ├──► Rule-based Aspect Tagging (Food / Service / Ambience / Price)
        └──► LDA Topic Modeling (Gensim, coherence-based k selection)
        │
        ▼
Streamlit Dashboard
```

One tuned Logistic Regression model handles both rating prediction and sentiment (derived from the rating). No separate Naive Bayes/BERT/Random Forest/XGBoost models are used — see Future Enhancements.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · NLTK · Gensim · Matplotlib · Seaborn · WordCloud · Streamlit · Joblib · Jupyter · Git

## Project Structure

```text
Local_Business_Review/
├── app/
│   ├── streamlit_app.py
│   └── text_utils.py
├── Data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── review_model.pkl
│   ├── lda_model.gensim (+ .id2word, .state, .expElogbeta.npy)
│   ├── lda_dictionary.gensim
│   └── topic_labels.json
├── notebooks/
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_sentiment_classification.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   └── 05_Topic_modeling.ipynb
├── src/
│   ├── merge_csv.py
│   └── preprocessing.py
├── scraper/
│   └── data_collection_notes.md
├── visualization/
│   ├── rating_distribution.png
│   ├── reviews_per_business.png
│   ├── review_length_distribution.png
│   └── wordcloud.png
├── Action_Plan.pdf
├── Local_Business_Review_Report.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation & Run

```bash
git clone https://github.com/Priyal9497/local-business-review-analyzer.git
cd local-business-review-analyzer
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet omw-1.4

python src/merge_csv.py                 # merge raw review sheets
python notebooks/01_data_cleaning.py    # clean + preprocess
python notebooks/02_eda.py              # EDA + visualizations

# train/evaluate model & topics (Jupyter)
jupyter notebook notebooks/03_sentiment_classification.ipynb
jupyter notebook notebooks/04_Model_Evaluation.ipynb
jupyter notebook notebooks/05_Topic_modeling.ipynb

streamlit run app/streamlit_app.py      # http://localhost:8501
```

## Dashboard Features

- Per-restaurant filter, review count, average rating, % positive, top aspect
- Rating and aspect distribution charts, average rating by restaurant
- Word cloud of review text

## Model Evaluation

Best hyperparameter: `C = 3.0` (GridSearchCV, C ∈ {0.5, 1, 3, 10}, 4-fold stratified CV, inner macro-F1 = 0.443).

**Rating prediction — held-out test (n = 116):**

| Rating | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 1★ | 0.45 | 0.42 | 0.43 | 12 |
| 2★ | 0.31 | 0.57 | 0.40 | 7 |
| 3★ | 0.29 | 0.20 | 0.24 | 10 |
| 4★ | 0.31 | 0.28 | 0.29 | 29 |
| 5★ | 0.68 | 0.69 | 0.68 | 58 |
| **Accuracy** | | | **0.51** | 116 |

Rating MAE (held-out): **0.675**

**Sentiment (derived from predicted rating) — held-out test:**

| Sentiment | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.62 | 0.84 | 0.71 | 19 |
| Neutral | 0.33 | 0.10 | 0.15 | 10 |
| Positive | 0.93 | 0.93 | 0.93 | 87 |
| **Accuracy** | | | **0.84** | 116 |

**5-fold CV (full dataset, n = 577):**

| Metric | Value |
|---|---:|
| Sentiment macro-F1 | 0.614 |
| Sentiment accuracy | 0.847 |
| Rating MAE | 0.662 |
| Rating RMSE | 0.855 |
| Rating R² | 0.577 |

## Limitations

- Single city, single platform, English-only
- Cleaned dataset skews toward 4–5★ despite stratified collection
- Neutral-sentiment and mid-range rating (2★–4★) classes are the weakest predicted
- Aspect tagging and topic labels are lexicon/rule-based, not learned
- No transformer-based model — not justified at this dataset size

## Future Enhancements

- Benchmark against Naive Bayes, BERT, Random Forest, XGBoost
- Aspect-based sentiment analysis (learned, not rule-based)
- Multilingual support; ordinal classification for ratings
- Explainability (SHAP/LIME); cloud deployment

## Ethics

Only public review text, ratings, and business names collected — no personal identifiers. Data used for academic analysis only.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Email : priyal31706@gmail.com
        rishabhsrv11@gmail.com