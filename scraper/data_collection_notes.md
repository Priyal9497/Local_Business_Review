# Data Collection — Restaurant Review Corpus

| | |
|---|---|
| **Project** | Restaurant Review Sentiment & Aspect Analyzer |
| **Module** | `scraper/` (data acquisition) |
| **Source platform** | [Google Maps / Zomato / JustDials] |
| **Raw output** | `Data/raw/` → merged into `Data/processed/merged_restaurant_reviews.csv` |
| **Version** | 1.0 |

---

## 1. Objective

Collect a labelled corpus of restaurant reviews — each record containing the
**review text**, the **star rating (1–5)**, and the **business name** — to train
and evaluate:

1. a rating-prediction and sentiment-classification model,
2. a rule-based aspect extractor (Food / Service / Ambience / …),
3. an LDA topic model for exploratory analysis.

Target size: ~10 restaurants × ~60 reviews each (≈ 600–700 reviews), balanced
across rating levels.

---

## 2. Data Source

Reviews were collected from the **publicly visible review pages** of 10
restaurants in Bengaluru on [platform name]:

> MTR, Vidyarthi Bhavan, Koko, The Only Place, Nagarjuna, Empire Restaurant,
> Meghana Foods, Paragon, Truffles, CTR Shri Sagar

---

## 3. Collection Methodology

### 3.1 Approach: Manual Curation (Human-in-the-Loop Collection)

The corpus was assembled **manually** by the author through the platform's
standard web interface. No automated crawler, bot, or scraping script was used.
This folder is named `scraper/` to match the project template's acquisition
stage; its deliverable is this methodology document and the curated dataset,
rather than crawler code.

### 3.2 Rationale — Why Manual Collection Over Automated Scraping

| Factor | Automated scraping | Manual curation (chosen) |
|---|---|---|
| **Terms of Service** | Review platforms explicitly prohibit automated extraction in their ToS; scraping would violate platform policy | Fully compliant — identical to normal human browsing |
| **Anti-bot defences** | CAPTCHAs, rate limits and dynamic loading make crawlers brittle and unreliable | Not applicable |
| **Sampling control** | Returns reviews in platform order (recency / "most relevant"), which is heavily skewed toward 4–5★ | Allowed **stratified sampling**: reviews were deliberately selected across all rating levels (1–5★), producing a balanced corpus suitable for classification |
| **Quality at source** | Requires heavy post-hoc cleaning (spam, duplicates, non-English, truncated text) | Low-quality entries were rejected *during* collection |
| **Scale** | Justified for 10⁴–10⁶ records | For a ~700-review academic corpus, manual collection is feasible and more reliable |

The decisive factor was **rating balance**. A scraped sample of these
restaurants would be ~70 % positive, making sentiment and rating models biased
toward the majority class. Manual stratified collection yielded roughly equal
representation of each star level per restaurant (see §4), which directly
improves model evaluation.

### 3.3 Step-by-Step Procedure

1. For each restaurant, opened its public review page on [platform].
2. Browsed reviews and, for each star level 1–5, copied the **full review
   text**, the **star rating**, and the **business name** into a structured
   spreadsheet (`Data/raw/`).
3. Targeted ≈ 60 reviews per restaurant, drawing evenly from each available
   rating level.
4. Skipped reviews that were empty, non-English, pure emoji, or obvious
   spam/ads.
5. Did **not** record reviewer names, profile IDs, or any personal
   information.
6. Exported each sheet to CSV and merged into
   `Data/processed/merged_restaurant_reviews.csv`.

### 3.4 Record Schema

| Column | Type | Description |
|---|---|---|
| `Business_Name` | string | Restaurant name |
| `Review text` | string | Full review text as displayed |
| `Rating` | int (1–5) | Star rating attached to the review |

---

## 4. Resulting Dataset

| Property | Value |
|---|---|
| Restaurants | 10 |
| Total reviews | [~625 — confirm with `len(df)`] |
| Reviews per restaurant | 60–65 (≈ 60) |
| Rating coverage | 1★–5★ per restaurant |
| Language | English |

The near-uniform per-business counts visible in the EDA bar chart are a direct
result of the stratified manual procedure.

---

## 5. Post-Collection Cleaning

Performed in the preprocessing notebook; the cleaned file is the one used by
all downstream models:

- Duplicate row removal.
- Whitespace normalisation on `Business_Name` (leading/trailing and
  non-breaking spaces) — this merged a duplicate entry caused by a
  trailing space during manual entry.
- Missing-value imputation (`Business_Name` → "Unlisted").
- Text normalisation for modelling (lower-casing, negation tagging) handled
  separately in `app/text_utils.py`.

---

## 6. Limitations & Bias

- **Selection bias:** manual choice of reviews is not perfectly random;
  mitigated by enforcing rating stratification per restaurant.
- **Geographic / platform scope:** single city (Bengaluru), single platform,
  English only — findings may not generalise to other cuisines, regions or
  languages.
- **Temporal scope:** reviews reflect the collection date; restaurant quality
  may have changed since.

---

## 7. Ethics & Privacy

- Only publicly visible content was recorded; no personal identifiers
  (reviewer names, avatars, IDs) were stored.
- The data is used exclusively for academic research and is not redistributed
  beyond this project.
- Collection respected the platform's Terms of Service by avoiding automated
  access entirely.

---

## 8. Reproducibility Checklist

To recreate the corpus from scratch:

- [ ] Open the public review page of each restaurant listed in §2.
- [ ] For each rating level 1–5, record ~12 reviews (text + rating + name).
- [ ] Apply the schema in §3.4 and export to `Data/raw/`.
- [ ] Run the preprocessing notebook to produce the merged, cleaned CSV.

---
