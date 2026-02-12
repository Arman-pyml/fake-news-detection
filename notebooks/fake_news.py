# STEP 1: Libraries
import pandas as pd

# STEP 2: Dataset load
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# STEP 3: Label add
fake['label'] = 0   # Fake
true['label'] = 1   # Real

# STEP 4: Combine dataset
data = pd.concat([fake, true])

# STEP 5: Data check
print("Dataset loaded successfully")
print(data.head())
import re

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

data['text'] = data['text'].apply(clean_text)

print("Text cleaning done")
import re

def clean_text(text):
    text = text.lower()                  # small letters
    text = re.sub('[^a-zA-Z]', ' ', text) # symbols remove
    return text

data['text'] = data['text'].apply(clean_text)

print("Text cleaning completed")

from sklearn.feature_extraction.text import TfidfVectorizer

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

print("Text converted into numerical form")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Train-Test split done")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model training completed")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_news(news):
    news = clean_text(news)
    vector = vectorizer.transform([news])
    result = model.predict(vector)
    return "REAL NEWS" if result[0] == 1 else "FAKE NEWS"

while True:
    user_news = input("\nEnter news (type exit to stop): ")
    if user_news.lower() == "exit":
        break
    print("Prediction:", predict_news(user_news))
