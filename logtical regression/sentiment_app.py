# sentiment_app.py
import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Twitter Sentiment Analyzer")

examples = {
    "Positive 😀": "I love this game!",
    "Negative 😠": "This is the worst app ever.",
    "Neutral 😐": "I installed it yesterday."
}

col1, col2, col3 = st.columns(3)
for i, (label, text) in enumerate(examples.items(), start=1):
    with eval(f"col{i}"):
        if st.button(label):
            st.session_state["comment"] = text

user_input = st.text_area("💬 Enter a comment:", value=st.session_state.get("comment", ""))

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter a comment.")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        label_map = {1: "Positive 😀", 0: "Neutral 😐", -1: "Negative 😠"}
        st.success(f"Predicted Sentiment: {label_map[pred]}")


