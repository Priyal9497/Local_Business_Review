import streamlit as st
import joblib
sentiment_model = joblib.load(open("models/sentiment_classifier.pkl", "rb"))

tfidf_vectorizer = joblib.load(open("models/tfidf_vectorizer.pkl", "rb"))
import gensim 

lda_model=gensim.models.LdaModel.load("models/lda_model.gensim")
dictionary=gensim.corpora.Dictionary.load("models/lda_dictionary.gensim")
st.sidebar.title("Navigation")
page=st.sidebar.selectbox(
    "Select a Page",
    ["Home","Visualization",
     "Sentiment Analysis","Rating Prediction","Topic Modeling"]
)

if page == "Home":
  st.title("Bengaluru Resturant Review Analyzer")
st.header("Project Overview")
st.write("""
This dashboard analyzes resturant reviews from Bengaluru
resturants.
It includes:
-Sentiment Analysis
-Rating Prediction
-Topic Modeling
Data Visualization
 """)
st.header("Data Visualization")
st.write("Resturant review analysis graphs will be displayed here.")
st.subheader("Rating Distribution")
st.image("visualization/rating_distribution.png")
st.subheader("Review Length Distribution")
st.image("visualization/review_length_distribution.png")
st.subheader("Reviews per Business")
st.image("visualization/reviews_per_business.png")
st.subheader("Word Cloud")
st.image("visualization/wordcloud.png")
st.header("Sentiment Analysis")
review=st.text_area("Enter a resturant review")
if st.button("Analyse Sentiment"):
  st.write("Sentiment result will be shown here.")
  if review.strip() == "":
    st.warning("Please enter a review.")
else:
    review_vector = tfidf_vectorizer.transform([review])
    prediction = sentiment_model.predict(review_vector)

    if prediction[0] == 1:
        st.success("😊 Positive Review")
    else:
        st.error("😞 Negative Review")
    st.header("Rating Prediction")
    review_for_rating=st.text_area("Enter review for rating prediction")
if st.button("Predict Rating"):
  st.write("Predict Rating will be shown here.")
st.header("Topic Modeling")
topic_review=st.text_area("Enter review for topic analysis")
if st.button("Find Topics"):
  
    if topic_review:
        words = topic_review.lower().split()

        bow = dictionary.doc2bow(words)

        topics = lda_model.get_document_topics(bow)

        st.write("Detected Topics:")
        st.write(topics)

    else:
        st.write("Please enter a review")
  
 
  

 
  