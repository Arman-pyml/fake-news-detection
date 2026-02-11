import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ======================
# Load Dataset
# ======================
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")



fake["label"] = 0   # Fake
true["label"] = 1   # Real

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# ======================
# Text Cleaning Function
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data["text"] = data["text"].apply(clean_text)

# ======================
# Train Test Split
# ======================
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ======================
# Vectorization
# ======================
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ======================
# Model Training
# ======================
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ======================
# Accuracy
# ======================
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ======================
# Save Model
# ======================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model & Vectorizer Saved")
