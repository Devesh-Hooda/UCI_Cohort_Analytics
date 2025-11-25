# -*- coding: utf-8 -*-
"""UCI_EDA_Analytics.ipynb

Original file is located at
    https://colab.research.google.com/drive/1AEwxbZrSDr_2_TwkAopdfGPARDWbQh2-

### 1. Data Loading and Initial Inspection

This section handles uploading the dataset and loading it into a pandas DataFrame. It then performs initial checks to understand the data's structure and contents.
"""

import os
from pathlib import Path
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.express as px
import plotly.graph_objects as go
import IPython.display as ipd  # use ipd.display(...) later

# add this block to enable display() in non-notebook runs
try:
    from IPython.display import display
except Exception:
    # simple fallback for scripts/IDE: print a small summary
    def display(obj):
        try:
            # try pandas-friendly output
            print(obj.head(12) if hasattr(obj, "head") else obj)
        except Exception:
            print(obj)

# Load the workbook from the repository relative path (use the requested filename)
data_file = Path(__file__).resolve().parent / "online_retail_II.xlsx"
if not data_file.exists():
    raise FileNotFoundError(f"Expected dataset at: {data_file}\nPlace the .xlsx file named 'online+retail+ii.xlsx' in the project folder.")

# read with openpyxl engine
df = pd.read_excel(data_file, engine="openpyxl")

# quick inspection (safe in scripts due to display fallback)
display(df.head())
df.info()
df.describe(include='all')
df.isnull().sum()
df.head(10)

#===============================================================================
# CREATE NEW COLUMNS AND CONVER EXISTING DATA COLUMNS

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create Revenue column
df['Revenue'] = df['Quantity'] * df['Price']

#===============================================================================

df.info()
df.head()
df.isnull().sum()

"""### 2. Data Pre-processing and validity

"""

# ---------------------------------------
# Drop rows with missing Customer ID
# ---------------------------------------

df_clean = df.dropna(subset=['Customer ID']).copy()

# Convert Customer ID to integer (remove decimal .0)
df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)

# ---------------------------------------
# Handle missing product descriptions
# ---------------------------------------
# If Description is missing, drop (very small percentage)

df_clean = df_clean.dropna(subset=['Description'])
df_clean['Description'] = df_clean['Description'].str.strip().str.upper() # Trim whitespace


# ---------------------------------------
# Handle negative quantities (returns)
# ---------------------------------------
# Remove negative quantities entirely

df_clean = df_clean[df_clean['Quantity'] > 0]

# ---------------------------------------
# Remove rows with invalid StockCodes
# Common invalid codes include: POST, D, C2, BANK CHARGES, etc.
# ---------------------------------------
invalid_codes = ['POST', 'D', 'C2', 'BANK CHARGES', 'M', 'DOT']
df_clean = df_clean[~df_clean['StockCode'].isin(invalid_codes)]

# ---------------------------------------
# Remove duplicate rows (if any)
# ---------------------------------------

df_clean = df_clean.drop_duplicates()

# ---------------------------------------
# Verify final datatypes
# ---------------------------------------

print(df_clean.info())
print(df_clean.head())

# ---------------------------------------
# Find the sum of null columns
# ---------------------------------------

print(df_clean.isnull().sum())

# Cell 1 — imports and base settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.express as px
import plotly.graph_objects as go

# Optional: nicer seaborn style for static plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12,6)

# Cell 2 — quick sanity checks (ensure df_clean exists)
print("Rows:", len(df_clean))
print("Date range:", df_clean['InvoiceDate'].min(), "to", df_clean['InvoiceDate'].max())
print("Total revenue (sum):", df_clean['Revenue'].sum())

# ---------------------------
# Monthly Revenue & Trend
# ---------------------------

# Create month column (period) and monthly aggregates
df_clean['InvoiceMonth'] = df_clean['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
monthly = df_clean.groupby('InvoiceMonth').agg({
    'Revenue': 'sum',
    'Invoice': pd.Series.nunique,   # number of orders (invoices)
    'Customer ID': pd.Series.nunique
}).rename(columns={'Invoice': 'NumOrders', 'Customer ID': 'NumCustomers'}).reset_index()

# Rolling mean for smoothing
monthly['Revenue_Rolling_3'] = monthly['Revenue'].rolling(3, min_periods=1).mean()

# Plot: Monthly Revenue + rolling
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(monthly['InvoiceMonth'], monthly['Revenue'], marker='o', label='Monthly Revenue')
ax.plot(monthly['InvoiceMonth'], monthly['Revenue_Rolling_3'], linestyle='--', label='3-month rolling')
ax.set_title("Monthly Revenue and 3-month Rolling Average")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (GBP)")
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fig = px.line(monthly, x='InvoiceMonth', y='Revenue', title='Monthly Revenue (interactive)', markers=True)
fig.add_scatter(x=monthly['InvoiceMonth'], y=monthly['Revenue_Rolling_3'], mode='lines', name='3M Rolling')
fig.update_xaxes(tickformat="%Y-%m")
fig.show()

# ---------------------------
# Top Products (by Revenue & Quantity)
# ---------------------------

# Product-level aggregations
prod_agg = df_clean.groupby(['StockCode','Description']).agg({
    'Revenue': 'sum',
    'Quantity': 'sum',
    'Invoice': pd.Series.nunique
}).rename(columns={'Invoice':'NumOrders'}).reset_index()

# Top 20 by Revenue
top20_rev = prod_agg.sort_values('Revenue', ascending=False).head(20)

# Top 20 by Quantity
top20_qty = prod_agg.sort_values('Quantity', ascending=False).head(20)


fig = px.bar(top20_rev,
             x='Revenue',
             y='Description',
             title='Top 20 Products by Revenue (Interactive)',
             orientation='h',
             labels={'Revenue': 'Total Revenue (GBP)', 'Description': 'Product Description'})
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

fig = px.bar(top20_qty,
             x='Quantity',
             y='Description',
             title='Top 20 Products by Quantity Sold (Interactive)',
             orientation='h',
             labels={'Quantity': 'Total Quantity Sold', 'Description': 'Product Description'})
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

# ---------------------------
# Cohort Analysis — Monthly Cohorts (Retention)
# ---------------------------

# 1) Identify each customer's first purchase month
df_clean['InvoiceMonth'] = df_clean['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
cust_first = df_clean.groupby('Customer ID')['InvoiceMonth'].min().reset_index()
cust_first.columns = ['Customer ID', 'CohortMonth']

# 2) Attach cohort to original df
df_cohort = df_clean.merge(cust_first, on='Customer ID')

# 3) Compute period number (months since cohort)
df_cohort['CohortIndex'] = ((df_cohort['InvoiceMonth'].dt.year - df_cohort['CohortMonth'].dt.year) * 12 +
                             (df_cohort['InvoiceMonth'].dt.month - df_cohort['CohortMonth'].dt.month)) + 1

# 4) Build cohort table: number of unique customers per cohort by period
cohort_counts = df_cohort.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().reset_index()
cohort_pivot = cohort_counts.pivot(index='CohortMonth', columns='CohortIndex', values='Customer ID')

# 5) Divide by cohort size to get retention rates
cohort_size = cohort_pivot.iloc[:,0]
retention = cohort_pivot.divide(cohort_size, axis=0).round(3)

# Display retention matrix (first 12 months)
display(retention.iloc[:,:12])

plt.figure(figsize=(12,8))
sns.heatmap(retention.iloc[:,:12], annot=True, fmt=".0%", cmap="YlGnBu")
plt.title("Cohort Retention Rate (by Cohort Month) — First 12 Periods")
plt.xlabel("Cohort Period (months since first purchase)")
plt.ylabel("Cohort Month")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------------------------
# Exports for Tableau
# ---------------------------

# Company summary (monthly)
monthly.to_csv('company_monthly_summary.csv', index=False)

# Product summary
prod_agg.to_csv('product_summary.csv', index=False)

# Cohort retention (export retention numeric values)
retention.reset_index().to_csv('cohort_retention.csv', index=False)

print("Exported: company_monthly_summary.csv, product_summary.csv, cohort_retention.csv, cleaned_online_retail.csv")

# Export the cleaned main dataset (no Colab downloads; files are saved to repo)
df_clean.to_csv("cleaned_online_retail.csv", index=False)

# Export monthly summary
monthly.to_csv("company_monthly_summary.csv", index=False)

# Export product-level summary
prod_agg.to_csv("product_summary.csv", index=False)

# Export cohort retention (retention matrix)
retention.reset_index().to_csv("cohort_retention.csv", index=False)

# Note: RFM table is exported after the RFM table is computed later in the script.

import pandas as pd
import numpy as np

# RECENCY REFERENCE DATE
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print("Snapshot Date:", snapshot_date)

# Group by Customer ID
rfm = df_clean.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Invoice':     'nunique',                                 # Frequency
    'Revenue':     'sum'                                      # Monetary
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm = rfm.reset_index()
rfm.head()

# Recency score (lower recency = better customer)
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)

# Frequency & Monetary (higher is better)
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)

rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
rfm.head()

def rfm_segment(row):
    r = row['R_score']
    f = row['F_score']
    m = row['M_score']

    if r >= 4 and f >= 4 and m >= 4:
        return 'Best Customers'
    if f >= 4 and m >= 3:
        return 'Loyal Customers'
    if r >= 3 and f >= 3:
        return 'Potential Loyalists'
    if r <= 2 and f <= 2:
        return 'At Risk'
    return 'Others'

rfm['Segment'] = rfm.apply(rfm_segment, axis=1)

rfm['Segment'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,3, figsize=(18,5))
sns.histplot(rfm['Recency'], ax=ax[0], kde=True)
sns.histplot(rfm['Frequency'], ax=ax[1], kde=True)
sns.histplot(rfm['Monetary'], ax=ax[2], kde=True)

ax[0].set_title("Recency Distribution")
ax[1].set_title("Frequency Distribution")
ax[2].set_title("Monetary Distribution")
plt.show()

rfm.to_csv("rfm_table.csv", index=False)
