import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("C:/Users/edlap/Documents/FInal year Project/Raja/Codsoft/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

# ===============================
# 5. Evaluate Model
# ===============================

predictions = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


def predict_sms(text):

    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    if prediction == 1:
        return "SPAM"
    else:
        return "HAM (Legitimate)"

while True:
    msg = input("Enter SMS: ")
    print(predict_sms(msg))
