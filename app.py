import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import shap
import joblib

st.set_page_config(layout="wide")
st.title("📊 FinIntel: Real-Time Anomaly Detection Dashboard")

# Load data
df = pd.read_csv("final_data.csv")
X = pd.read_csv("X_features.csv")
explainer = joblib.load("explainer.pkl")

# ⏱️ Key Metrics
st.subheader("📌 Key Anomaly Stats")
col1, col2 = st.columns(2)
col1.metric("⚠️ Total Anomalies", df["AnomalyLabel"].value_counts().get("Anomaly", 0))
col2.metric("✅ Normal Clients", df["AnomalyLabel"].value_counts().get("Normal", 0))

# 🔍 Filters
st.subheader("🎛️ Client Filters")

client = st.selectbox("🔍 Select a Client ID", df["ClientID"].unique())
st.write(df[df["ClientID"] == client])

loan_filter = st.selectbox("🎯 Filter by Loan Purpose", ["All"] + list(df["LoanPurpose"].unique()))
if loan_filter != "All":
    st.write(df[df["LoanPurpose"] == loan_filter])

# 📈 Credit Score Distribution
st.subheader("📈 Credit Score Distribution")
fig, ax = plt.subplots()
sns.histplot(data=df, x="CreditScore", hue="AnomalyLabel", kde=True, ax=ax)
st.pyplot(fig)

# 🔍 Detected Anomalies
st.subheader("🧨 All Detected Anomalies")
st.dataframe(df[df["AnomalyLabel"] == "Anomaly"])

# 🗃️ SQL: High-Risk Clients
st.subheader("📄 High-Risk Clients from SQL (CreditScore < 600)")
conn = sqlite3.connect("finintel.db")
query = """
SELECT ClientID, Age, Income, LoanAmount, CreditScore
FROM applications
WHERE AnomalyLabel = 'Anomaly' AND CreditScore < 600
"""
high_risk_df = pd.read_sql(query, conn)
conn.close()
st.dataframe(high_risk_df)
