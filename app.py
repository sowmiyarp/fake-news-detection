import streamlit as st
import joblib

import os

st.write("Current working directory:", os.getcwd())
st.write("Files in this directory:", os.listdir())
# Load saved model and vectorizer
try:
    model = joblib.load(os.path.join(os.getcwd(),"model.pkl"))

    vectorizer = joblib.load("tfidf.pkl")
    st.success("Model and Vectorizer loaded successfully.")
except Exception as e:
    st.error(f" Error loading model/vectorizer: {e}")
    st.stop()

# Function to predict
def predict_news(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0][pred]
    label = "🟢 Real News" if pred == 1 else "🔴 Fake News"
    return label, prob

# Streamlit UI
st.set_page_config(page_title="Fake News Detector")
st.title(" Fake News Detection App")

user_input = st.text_area("Enter a news article or paragraph here:", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence = predict_news(user_input)
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}")


