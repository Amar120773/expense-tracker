# model/train.py

import pandas as pd
import joblib
import mlflow

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("data/expenses.csv")

print("Dataset Loaded ✅")
print(df.head())

# =========================
# 2. Preprocessing
# =========================
df['description'] = df['description'].str.lower()

X = df['description']
y = df['category']

# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Feature Extraction (TF-IDF)
# =========================
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 5. Model Training
# =========================
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# =========================
# 6. Evaluation
# =========================
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 7. MLflow Tracking
# =========================
mlflow.start_run()

mlflow.log_param("model", "LogisticRegression")
mlflow.log_metric("accuracy", accuracy)

mlflow.end_run()

# =========================
# 8. Save Model
# =========================
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n✅ Model and vectorizer saved!")
