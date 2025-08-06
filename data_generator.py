import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import joblib
import sqlite3

# Simulate 100 clients
np.random.seed(42)
client_ids = [f"CL-{1000+i}" for i in range(100)]

# Structured + unstructured data
data = {
    "ClientID": client_ids,
    "Age": np.random.randint(21, 65, size=100),
    "Income": np.random.randint(25000, 150000, size=100),
    "LoanAmount": np.random.randint(1000, 50000, size=100),
    "CreditScore": np.random.randint(300, 850, size=100),
    "LoanPurpose": np.random.choice(
        ["Home", "Car", "Business", "Education", "Medical", "Travel"], size=100
    ),
    "ApplicationDescription": np.random.choice([
        "Need urgent funds due to surgery",
        "Starting a new business venture",
        "College fees payment required soon",
        "Buying a second-hand vehicle",
        "Planning family vacation",
        "House renovation loan requested"
    ], size=100)
}

df = pd.DataFrame(data)

# TF-IDF on descriptions
tfidf = TfidfVectorizer(max_features=5)
tfidf_matrix = tfidf.fit_transform(df["ApplicationDescription"]).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf.get_feature_names_out())
df = pd.concat([df, tfidf_df], axis=1)

# Isolation Forest Anomaly Detection
features = ["Age", "Income", "LoanAmount", "CreditScore"] + list(tfidf.get_feature_names_out())
model = IsolationForest(contamination=0.1, random_state=42)
df["Anomaly"] = model.fit_predict(df[features])
df["AnomalyLabel"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

# SHAP Explainer
df["is_anomaly"] = (df["AnomalyLabel"] == "Anomaly").astype(int)
X = df[features]
clf = RandomForestClassifier(random_state=42)
clf.fit(X, df["is_anomaly"])
explainer = shap.Explainer(clf, X)

# Save all outputs
df.to_csv("final_data.csv", index=False)
X.to_csv("X_features.csv", index=False)
joblib.dump(explainer, "explainer.pkl")

# SQL write (for ION-style analytics)
conn = sqlite3.connect("finintel.db")
df.to_sql("applications", conn, if_exists="replace", index=False)
conn.close()
