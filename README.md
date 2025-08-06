# 📊 FinIntel: Real-Time Anomaly Detection & Risk Analysis

**Detect financial anomalies and assess loan risk in real-time using structured + unstructured data, all inside an interactive Streamlit dashboard.**

---

## 🚀 Features

- 🧠 **Anomaly Detection** using Isolation Forest on client loan & credit data
- ✍️ **TF-IDF NLP** on application descriptions for added fraud insights
- 🧰 **Real-Time Dashboard** built in Streamlit with filtering and stats
- 💾 **SQLite Integration** for dynamic SQL querying of high-risk clients
- 📈 **KPI Cards + Distributions** for credit risk visualization

---

## 🛠 Tech Stack

- Python, pandas, scikit-learn, seaborn, matplotlib
- Streamlit for dashboard
- SQLite for data query pipeline
- IsolationForest for anomaly detection
- TF-IDF from scikit-learn for text vectorization

---

## 🧪 Sample Use Case

> “Given 100 loan applications, which ones look suspicious and why?”

This dashboard enables analysts to:
- Spot anomalies (e.g. low credit, high loan) visually
- Filter by loan purpose or client ID
- Query high-risk clients via SQL (e.g., credit < 600)

---

## ✅ Try It Out

```bash
# Clone and install
python data_generator.py
# Run dashboard
streamlit run app.py
