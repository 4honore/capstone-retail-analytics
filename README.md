
# 🛒 Retail Customer Analytics – Capstone Project (INSY 8413)

**Student:** ISHIMWE Honore  
**ID:** 26578  
**Course:** Introduction to Big Data Analytics (INSY 8413)  
**Semester:** III, Academic Year 2024–2025  
**Instructor:** Mr. Eric Maniraguha  

---

## 📌 Table of Contents

- [📌 Problem Statement](#-Problem-Statement)
- [📘 Project Overview](#-project-overview)
- [📊 Dataset Summary](#-dataset-summary)
- [🧪 Python Data Analytics](#-python-data-analytics)
  - [🔧 1. Data Cleaning](#1-data-cleaning)
  - [📊 2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [📈 3. RFM Clustering](#3-rfm-clustering)
  - [🧠 4. Churn Prediction (Innovation)](#4-churn-prediction-innovation)
- [📉 Model Evaluation](#-model-evaluation)
- [📊 Power BI Dashboard](#-power-bi-dashboard)
- [💡 Key Insights](#-key-insights)
- [🛠️ Project Files Structure](#-project-files-structure)
- [🚀 Future Work](#-future-work)
- [🖼️ Screenshots](#-screenshots)
- [📬 Submission & Contact](#-submission--contact)
- [📢 Academic Integrity](#-academic-integrity)

---

## 📌 Problem Statement
Retail businesses often struggle to understand their diverse customer base, leading to inefficient marketing, poor retention, and lost revenue. Without data-driven insights into customer behavior, it's challenging to design effective loyalty programs or prioritize high-value segments.

**This project solves that by using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to segment customers.** It also introduces an innovative churn prediction model to identify at-risk customers based on their RFM scores, helping businesses take proactive retention actions.

---

## 📘 Project Overview

This capstone project focuses on customer segmentation and churn prediction for a retail business using the **Online Retail Dataset**. The project involves:

- Data preprocessing and cleaning  
- RFM analysis and customer clustering using **K-Means**
- Predicting customer churn using **Decision Tree Classifier**
- Building a **Power BI dashboard** to present key business insights

---

## 📊 Dataset Summary

- **Dataset Name:** Online Retail  
- **Source:** UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/352/online+retail  
- **Type:** Structured Excel file  
- **Size:** ~540,000 rows × 8 columns  
- **Attributes:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country  
- **Preprocessing Steps:** 
  - Removed nulls in `CustomerID`  
  - Removed negative and zero values in `Quantity` and `UnitPrice`  
  - Converted `InvoiceDate` to datetime  
  - Added `TotalPrice = Quantity × UnitPrice`  
  - Removed outliers using IQR method

---

## 🧪 Python Data Analytics

All Python analysis was performed in **Jupyter Notebook** using `pandas`, `sklearn`, `seaborn`, and `matplotlib`.

---

### 🔧 1. Data Cleaning

```python
def clean_data(df):
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    for col in ['Quantity', 'TotalPrice']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df
```

---

### 📊 2. Exploratory Data Analysis (EDA)

```python
def perform_eda(df):
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['TotalPrice'].sum()
    monthly_sales.plot(kind='line', title='Monthly Sales Trend')

    country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(5)
    sns.barplot(x=country_sales.values, y=country_sales.index)

    sns.histplot(df['Quantity'], bins=30)
```

Key Findings:
- UK dominates in revenue
- Monthly sales peak in November and December
- Most transactions are of small to moderate quantity

---

### 📈 3. RFM Clustering

```python
def compute_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
    return rfm
```

```python
def train_kmeans(rfm, n_clusters=4):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm, kmeans
```

---

### 🧠 4. Churn Prediction (Innovation)

```python
def predict_churn(rfm):
    rfm['Churn'] = (rfm['Recency'] > 180).astype(int)
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    rfm['Churn_Prediction'] = model.predict(X)
    return rfm
```

---

## 📉 Model Evaluation

- **Silhouette Score**: Evaluated cluster quality  
- **Decision Tree Accuracy**: ~85% (based on holdout set)  
- RFM & churn predictions exported to CSV for Power BI

---

## 📊 Power BI Dashboard

Power BI was used to present insights visually.

Key Features:
- Monthly trend chart  
- Map showing sales by country  
- RFM scatter plot  
- Filter by cluster, churn status, country  
- KPI cards for total sales and customers

File: `capstone_dashboard.pbix`

---

## 💡 Key Insights

- UK is the top customer base  
- Cluster 0 customers are least valuable  
- High Recency customers show higher churn risk  
- November is the best-performing month  

---

## 🛠️ Project Files Structure

```
capstone-retail-analytics/
├── cleaned_retail_data.csv
├── rfm_clusters.csv
├── rfm_churn_prediction.csv
├── Online_Retail.ipynb
├── capstone_dashboard.pbix
├── presentation.pptx
├── screenshots/
│   ├── dashboard_overview.png
│   ├── monthly_sales.png
│   ├── rfm_clusters.png
│   └── churn_prediction.png
└── README.md
```

---

## 🚀 Future Work
  
- Use market basket analysis for product patterns  
- Connect dashboard to live retail system

---

## 🖼️ Screenshots


- `dashboard_overview.png`: Full dashboard
  <img width="1366" height="768" alt="dashboard1" src="https://github.com/user-attachments/assets/c259b8eb-0bca-4e9c-8664-4482809d6f5e" />

- `churn_prediction.png`: Predicted churn customers
  
<img width="1366" height="423" alt="Churn_Prediction_Output" src="https://github.com/user-attachments/assets/aa50aabc-1816-4e4a-b461-c3bec7e1f893" />

---

## 📬 Submission & Contact

**Author:** ISHIMWE Honore   
**Instructor:** Mr. Eric Maniraguha  
📩 Email: [eric.maniraguha@auca.ac.rw](mailto:eric.maniraguha@auca.ac.rw)

---

## 📢 Academic Integrity

> I affirm that this project is my own original work.  
> Any external resources were used responsibly and ethically.  

“Whatever you do, work at it with all your heart, as working for the Lord…” – *Colossians 3:23*
