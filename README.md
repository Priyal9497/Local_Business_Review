# Local Business Review Analyzer
## Sentiment Analysis, Rating Prediction, Topic Modeling, and Business Intelligence from Customer Reviews

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20BERT-orange.svg)](https://scikit-learn.org/)

---

## Overview

The **Local Business Review Analyzer** is a Data Science and Natural Language Processing (NLP) project that analyzes manually collected customer reviews from local businesses such as restaurants, cafés, salons, retail stores, clinics, gyms, and service providers.

The system provides key insights by:

* Classifying customer sentiment
* Predicting star ratings from review text
* Identifying common topics using topic modelling
* Performing exploratory data analysis (EDA)
* Presenting results through an interactive Streamlit dashboard

The project helps business owners understand customer satisfaction, identify common issues, monitor public perception, and make informed business decisions.

---

## Problem Statement

Local businesses receive valuable feedback through online customer reviews. However, manually reading and interpreting a large number of reviews is difficult, time-consuming, and inconsistent.Average ratings alone do not explain why customers are satisfied or dissatisfied. For example, a business may receive low ratings due to slow service, poor staff behavior, high prices, product quality issues, or cleanliness concerns.
This project addresses this problem by using Data Analytics, Machine Learning, Artificial Intelligence, and NLP techniques to automatically analyze review text and generate useful business intelligence.

---

## Project Objectives

The main objectives of this project are:

1. Collect and organize customer reviews from local businesses.
2. Clean and preprocess customer review text.
3. Perform exploratory data analysis on ratings and review patterns.
4. Classify reviews into positive, negative, and neutral sentiment categories.
5. Predict customer star ratings based on review content.
6. Identify key review topics using topic modeling.
7. Visualize insights using charts, word clouds, and an interactive dashboard.
8. Help businesses identify strengths, weaknesses, and improvement opportunities.

---

## Key Features

- Manual customer review data collection
- Text cleaning and NLP preprocessing
- Sentiment classification
- Rating prediction
- Topic modeling using Latent Dirichlet Allocation (LDA)
- TF-IDF text vectorization
- Comparison of multiple Machine Learning models
- Business-focused visualizations
- Interactive Streamlit dashboard
- Reproducible project workflow using Git and GitHub

---

## Technologies Used

### Programming Language
- Python 3.9+

### Data Analysis and Processing
- Pandas
- NumPy

### Natural Language Processing
- NLTK
- Gensim
- Scikit-learn

### Machine Learning
- Scikit-learn
- XGBoost
- BERT / Transformer models

### Data Visualization
- Matplotlib
- Seaborn
- WordCloud

### Development and Deployment
- Jupyter Notebook
- Streamlit
- Git
- GitHub

---

## Project Workflow

```text
Manual Review Collection
        │
        ▼
Data Cleaning and Validation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Text Preprocessing
        │
        ▼
Feature Engineering and TF-IDF Vectorization
        │
        ├──► Sentiment Classification
        │       ├── Logistic Regression
        │       ├── Multinomial Naive Bayes
        │       └── BERT (Optional Advanced Model)
        │
        ├──► Rating Prediction
        │       ├── Linear Regression
        │       ├── Random Forest Regressor
        │       └── XGBoost Regressor
        │
        └──► Topic Modeling
                └── Latent Dirichlet Allocation (LDA)
        │
        ▼
Evaluation and Visualization
        │
        ▼
Streamlit Business Intelligence Dashboard
```

---

## Dataset Description

The project uses a manually collected dataset of customer reviews from local businesses.

Each review record may include the following fields:

| Column Name | Description |
|---|---|
| `review_id` | Unique identifier for each review |
| `business_name` | Name of the local business |
| `business_category` | Type of business, such as restaurant, salon, café, or clinic |
| `review_text` | Written feedback provided by the customer |
| `rating` | Customer rating, generally from 1 to 5 |
| `location` | General city or location, if available |
| `sentiment_label` | Sentiment category: Positive, Negative, or Neutral |

### Sample Dataset Format

```csv
review_id,business_name,business_category,review_text,rating,review_date,location,sentiment_label
1,ABC Cafe,Restaurant,"The food was delicious and the staff were very friendly.",5,2024-01-15,City A,Positive
2,ABC Cafe,Restaurant,"Service was very slow and the food arrived cold.",2,2024-01-18,City A,Negative
3,XYZ Salon,Salon,"The haircut was good but I had to wait for a long time.",3,2024-02-01,City B,Neutral
```

> **Important:** Personally identifiable information such as customer names, usernames, profile links, phone numbers, and email addresses should not be included in the final dataset.

---

## Sentiment Labeling Strategy

| Rating | Sentiment Label |
|---|---|
| 4–5 Stars | Positive |
| 3 Stars | Neutral |
| 1–2 Stars | Negative |

A manual review of a sample of labeled records is recommended because review text and star ratings may occasionally be inconsistent. For example:

- A customer may give a 4-star rating but mention a serious complaint.
- A customer may give a 2-star rating while appreciating one aspect of the service.
- A customer may use sarcasm or mixed sentiments in a review.

---

## NLP Preprocessing Pipeline

### Preprocessing Steps

1. Convert text to lowercase.
2. Remove missing or duplicate reviews.
3. Remove URLs, HTML tags, and unwanted symbols.
4. Remove extra spaces and irrelevant punctuation.
5. Tokenize text into individual words.
6. Remove stop words where appropriate.
7. Preserve important negation words such as `not`, `no`, and `never`.
8. Apply lemmatization to convert words to their root forms.
9. Create clean text features for machine-learning models.
10. Use BERT tokenizer separately when using transformer models.

### Example

**Original Review:**
```text
"The staff were not helpful, and the service was extremely slow!!!"
```

**Cleaned Review:**
```text
"staff not helpful service extremely slow"
```

---

## Machine Learning Models

### 1. Sentiment Classification Models

| Model | Description |
|---|---|
| Logistic Regression | Strong and interpretable baseline model for text classification |
| Multinomial Naive Bayes | Fast probabilistic model suitable for text data |
| BERT | Advanced transformer model that understands contextual meaning |

#### Sentiment Classes
- Positive
- Neutral
- Negative

#### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

### 2. Rating Prediction Models

| Model | Description |
|---|---|
| Linear Regression | Baseline regression model |
| Random Forest Regressor | Ensemble model that captures nonlinear patterns |
| XGBoost Regressor | Gradient boosting model for advanced prediction performance |

#### Rating Range
```text
1 Star = Very Dissatisfied
2 Stars = Dissatisfied
3 Stars = Neutral / Average
4 Stars = Satisfied
5 Stars = Very Satisfied
```

#### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared Score (R²)

---

### 3. Topic Modeling

Latent Dirichlet Allocation (LDA) is used to identify hidden themes in customer reviews.

| Topic Area | Example Keywords |
|---|---|
| Customer Service | staff, manager, friendly, rude, helpful |
| Product Quality | delicious, fresh, quality, taste, product |
| Price and Value | expensive, affordable, price, value, cost |
| Waiting Time | slow, queue, wait, delay, appointment |
| Cleanliness and Ambience | clean, parking, seating, atmosphere, hygiene |

---

## Visualizations

- Distribution of customer ratings
- Positive, neutral, and negative sentiment distribution
- Most frequent words in positive reviews
- Most frequent words in negative reviews
- Word clouds for positive and negative sentiment
- Feature importance for rating prediction models
- Topic distribution chart
- Business category comparison
- Average rating by business or category

---

## Suggested Project Structure

```text
Local_Business_Review/
├── app/
│   ├── streamlit_app.py
│   └── text_utils.py
│
├── Data/
│   ├── processed/
│   └── raw/
│
├── models/
│   ├── lda_dictionary.gensim
│   ├── lda_model.gensim
│   ├── lda_model.gensim.expElogbeta.npy
│   ├── lda_model.gensim.id2word
│   ├── lda_model.gensim.state
│   ├── review_model.pkl
│   └── topic_labels.json
│
├── notebooks/
|   |__ 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_sentiment_classification.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   └── 05_Topic_modeling.ipynb
│
├── scraper/
│   └── data_collection_notes.md
│
├── src/
│   ├── merge_csv.py
│   ├── preprocessing.py
|
|── visualization.py/
|   |── rating_distribution.png
|   ├── review_length_distribution.png
|   ├── reviews_per_business.png
|   ├── wordcloud.png
|
├── .gitignore
├── Action_Plan.pdf
├── Local_Business_Review_Report.pdf
├── README.md
├── requirements.txt
```
---

## Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/local-business-review-analyzer.git
cd local-business-review-analyzer
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Download Required NLTK Resources
```bash
python -m nltk.downloader punkt stopwords wordnet omw-1.4
```
---

## How to Run the Project

### Step 1: Add Dataset
Place the manually collected review dataset in:
```text
data/raw/customer_reviews.csv
```

### Step 2: Run Data Preprocessing
```bash
python src/preprocessing.py
```

### Step 3: Train Sentiment Classification Models
```bash
python notebooks/sentiment_model.ipynb
```

### Step 4: Train Rating Prediction Models
```bash
python notebooks/rating_prediction_model.ipynb
```

### Step 5: Run Topic Modeling
```bash
python src/topic_modeling.py
```

### Step 6: Launch the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

After running the command, open the displayed local URL in a web browser, usually:
```text
http://localhost:8501
```

---

## Streamlit Dashboard Features

### Dashboard Overview
- Total number of reviews
- Average business rating
- Positive sentiment percentage
- Negative sentiment percentage
- Most discussed customer topic

### Sentiment Analysis Page
- Upload review dataset
- View sentiment distribution
- Predict sentiment from custom review text
- Display classification probabilities
- Review positive and negative keywords

### Rating Prediction Page
- Enter a customer review
- Predict expected rating
- Compare actual and predicted ratings
- Display model performance metrics

### Topic Modeling Page
- Display discovered topics
- Show topic keywords
- Identify common customer concerns
- Compare topics across business categories

### Business Intelligence Page
- Rating trends over time
- Customer satisfaction insights
- Service quality analysis
- Common complaints and recommendations

---

## Example Business Insights

- Customers appreciate product quality but complain about service delays.
- Negative reviews frequently mention rude staff or poor customer support.
- High ratings are associated with words such as "friendly," "excellent," "clean," and "recommended."
- Lower ratings are associated with words such as "slow," "rude," "expensive," "dirty," and "disappointed."
- Customer satisfaction varies by business category, location, or time period.
- Waiting time and pricing may be major factors influencing negative ratings.

---

## Model Evaluation

> Metrics below come from the held-out test set (n = 116) and 5-fold stratified cross-validation, as run in `notebooks/04_Model_Evaluation.ipynb`. Best hyperparameter: `C = 3.0` (inner CV macro-F1: 0.443, GridSearchCV over `C ∈ {0.5, 1, 3, 10}`, 4-fold stratified CV).

### Rating Prediction — Held-Out Test Set

| Rating | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 1★ | 0.45 | 0.42 | 0.43 | 12 |
| 2★ | 0.31 | 0.57 | 0.40 | 7 |
| 3★ | 0.29 | 0.20 | 0.24 | 10 |
| 4★ | 0.31 | 0.28 | 0.29 | 29 |
| 5★ | 0.68 | 0.69 | 0.68 | 58 |
| **Accuracy** | | | **0.51** | 116 |
| **Macro avg** | 0.41 | 0.43 | 0.41 | 116 |
| **Weighted avg** | 0.51 | 0.51 | 0.50 | 116 |

**Rating MAE (held-out): 0.675** | Predicted range: 1.35–4.93

### Sentiment (Derived from Predicted Rating) — Held-Out Test Set

| Sentiment | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.62 | 0.84 | 0.71 | 19 |
| Neutral | 0.33 | 0.10 | 0.15 | 10 |
| Positive | 0.93 | 0.93 | 0.93 | 87 |
| **Accuracy** | | | **0.84** | 116 |
| **Macro avg** | 0.63 | 0.62 | 0.60 | 116 |
| **Weighted avg** | 0.83 | 0.84 | 0.83 | 116 |

### 5-Fold Cross-Validation (full dataset, n = 577)

| Metric | Value |
|---|---:|
| Sentiment macro-F1 | 0.614 |
| Sentiment accuracy | 0.847 |
| Rating MAE | 0.662 |
| Rating RMSE | 0.855 |
| Rating R² | 0.577 |
| Predicted rating range | 1.03 – 4.94 |

> Naive Bayes, BERT, Random Forest, and XGBoost were not implemented — see [Future Enhancements](#future-enhancements) for planned benchmarking against these models.

---

## Ethical Considerations

- Collect only publicly available or authorized review data.
- Do not collect personally identifiable information.
- Remove usernames, phone numbers, email addresses, and profile links.
- Respect review-platform terms and conditions.
- Use customer review data only for academic, research, or approved business-analysis purposes.
- Avoid presenting model predictions as absolute truth.
- Review models for bias across business categories, locations, or writing styles.
- Ensure that negative reviews are used constructively to improve services rather than unfairly target individuals.

---

## Limitations

- Manually collected datasets may be relatively small.
- Review ratings may not always match written sentiment.
- Sarcasm, irony, slang, and mixed opinions can be difficult to classify.
- BERT requires more computational resources than traditional models.
- Topic modeling may generate topics that need manual interpretation.
- Rating prediction may be affected by imbalanced rating distributions.
- Reviews may contain spelling errors, multiple languages, emojis, and informal writing styles.

---

## Future Enhancements

- Multilingual sentiment analysis
- Aspect-based sentiment analysis
- Naive Bayes, BERT, Random Forest, and XGBoost will implement.
- Real-time review monitoring through approved APIs
- Review summarization using Large Language Models
- Customer complaint alert system
- Location-based sentiment comparison
- Recommendation system for business improvement
- Deep-learning models such as LSTM, GRU, or RoBERTa
- Ordinal classification for 1–5 star ratings
- Explainable AI methods such as SHAP or LIME
- Deployment to cloud platforms such as Streamlit Cloud, Render, AWS, or Hugging Face Spaces

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Scikit-learn for Machine Learning tools
- NLTK for NLP preprocessing support
- Gensim for topic modeling
- Hugging Face for transformer-based NLP models
- Streamlit for dashboard development
- Matplotlib and Seaborn for data visualization
- Open-source Python community for development resources

---

## Contact

For questions, suggestions, or collaboration:

- **GitHub:** [https://github.com/Priyal9497/Local_Business_Review](https://github.com/Priyal9497/Local_Business_Review)

---