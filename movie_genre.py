import pandas as pd
import csv
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r"C:/Users/edlap/Documents/FInal year Project/Raja/Codsoft/train_data.txt",
    sep="::",
    engine="python",
    header=None,
    quoting=csv.QUOTE_NONE,
    on_bad_lines="skip"
)

df.columns = ["id", "movie", "genre", "summary"]

df = df[["summary", "genre"]]

df["genre"] = df["genre"].str.replace(":", "")
df["genre"] = df["genre"].str.strip()

df.dropna(inplace=True)

print("Dataset shape:", df.shape)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["summary"] = df["summary"].apply(clean_text)

X = df["summary"]
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1,2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB(alpha=0.5)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n🎬 Movie Genre Prediction System")

while True:
    plot = input("\nEnter a movie plot (or type 'exit' to stop): ")

    if plot.lower() == "exit":
        break

    if plot.strip() == "":
        print("Please enter a valid plot.")
        continue

    plot = clean_text(plot)

    plot_vec = vectorizer.transform([plot])

    prediction = model.predict(plot_vec)

    print("Predicted Genre:", prediction[0])
