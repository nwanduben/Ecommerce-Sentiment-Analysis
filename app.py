import streamlit as st
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("E-commerce Sentiment Analysis")
st.write("Enter a product review, and I'll predict the sentiment!")

user_input = st.text_area("Enter a review:")
if st.button("Analyze Sentiment"):
    if user_input:
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Neutral 😐" if prediction == 0 else "Negative 😠"
        st.success(f"Sentiment: {sentiment}")
