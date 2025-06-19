# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\ADMIN\Documents\liner\dataset\twitter_dataset\twitter_training.csv")
df.columns = ["id", "game", "sentiment", "comments"]
df = df[["sentiment", "comments"]].dropna()

# Clean data
df["sentiment"] = df["sentiment"].str.strip().str.lower()
df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]
df["sentiment"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["comments"])
y = df["sentiment"]

# Train model
model = LogisticRegression(max_iter=7000)
model.fit(X, y)

# Save both
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and Vectorizer saved!")
