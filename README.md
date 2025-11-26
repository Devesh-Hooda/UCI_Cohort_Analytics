# UCI Cohort Analytics

## 1) Executive summary
This repository contains a reproducible analysis of the UCI Online Retail dataset to measure revenue performance, product contribution, and customer behavior. Key analyses include monthly revenue trends, top product identification, cohort retention analysis, and RFM (Recency–Frequency–Monetary) segmentation. The dataset was cleaned to remove invalid records and exports are produced for downstream visualization in Tableau.

Snapshot findings:
- Clean dataset: ~399,643 transactions
- Analysis period: Dec 2009 — Dec 2010
- Total revenue (approx): £8.64M

## 2) Business Problem
E-commerce stakeholders need clear, actionable insights to:
- Track revenue trends and detect seasonality or declines
- Identify top-performing products and inventory priorities
- Understand customer retention and churn patterns by cohort
- Segment customers for targeted marketing and retention strategies

The goal of this project is to turn raw transaction data into scalable metrics and exports that support marketing, product, and operations decisions.

## 3) Methodology
Data processing and analytical steps implemented in the notebook/script:
- Environment and tools:
  - Python (pandas, numpy), plotting (matplotlib, seaborn, plotly), and optional Tableau for dashboarding.
- Data ingestion:
  - Read Excel/CSV transaction data into pandas DataFrame.
- Data cleaning:
  - Remove records with missing Customer ID or Description.
  - Convert Customer ID to integer, parse InvoiceDate to datetime.
  - Remove negative quantities (returns) and non-product StockCode items (e.g., POST, BANK CHARGES).
  - Deduplicate records and compute Revenue = Quantity * Price.
- Aggregations and analysis:
  - Monthly revenue aggregation and 3-month rolling trend.
  - Product-level aggregations (Revenue, Quantity, Number of Orders) and top-N listings.
  - Cohort analysis:
    - Define cohort by each customer's first purchase month.
    - Compute monthly retention matrix and visualize with heatmap.
  - RFM segmentation:
    - Compute Recency, Frequency, Monetary metrics (snapshot date = last invoice + 1 day).
    - Score customers into quintiles and assign segments (Best Customers, Loyal, Potential Loyalists, At Risk, Others).
- Outputs:
  - CSV exports for Tableau: company_monthly_summary.csv, product_summary.csv, cohort_retention.csv, rfm_table.csv, plus cleaned transaction export.

## 4) Result and Business Recommendation
Results (high-level)
- Revenue: Monthly revenue visualizations reveal fluctuations; smoothing with a 3-month rolling average aids trend interpretation.
- Products: A small set of SKUs drives disproportionate revenue and quantity — prioritise inventory and promotions for these items.
- Retention: Cohort heatmaps show retention decay over time; many cohorts lose a substantial share of buyers after the initial months.
- RFM: Customers segment naturally into high-value and at-risk groups enabling targeted engagement.

Recommendations
- Retention programs: Implement targeted reactivation campaigns for cohorts with steep drop-offs (e.g., email offers within months 1–3).
- Loyalty focus: Prioritize "Best Customers" and "Loyal Customers" with VIP offers and early product access to increase lifetime value.
- Product strategy: Optimize inventory and merchandising for top-performing SKUs; consider cross-sell bundles with top items.
- Data quality & pipeline: Automate ingestion and validation to prevent bad StockCodes, missing Customer IDs, and negative quantities from skewing reporting.
- Dashboarding: Publish the exported CSVs to a Tableau dashboard for ongoing monitoring and stakeholder access.

## 5) Further steps
Suggested follow-ups to increase impact and robustness:
- Build automated ETL to validate and refresh data on a schedule (Azure/AWS/GCP + Airflow or cron).
- Extend cohort analysis to lifetime value (LTV) per cohort and per segment.
- Run uplift/A-B tests on retention offers to quantify ROI of reactivation campaigns.
- Add product affinity analysis (market basket) for personalized recommendations and bundling.
- Implement unit tests for key data transformations and CI checks to prevent regressions.
- Deploy a lightweight web dashboard (Streamlit/Flask) for non-Tableau users to explore RFM and cohorts interactively.

How to run (quick)
1. Install dependencies (example):
   pip install pandas numpy matplotlib seaborn plotly openpyxl
   Note: Use a virtual environment (e.g., venv) to manage dependencies. The script requires openpyxl for reading Excel files.
2. Place the raw dataset in the project directory (correct filename: online_retail_II.xlsx, with Excel format).
3. Run the analysis script/notebook:
   - For script: python uci_eda_analytics.py
     Caution: Matplotlib and Plotly plots in script mode may not display in console environments without GUI support and could cause the script to hang. For console execution, consider modifying the script to save plots instead (e.g., plt.savefig() and fig.write_html()).
   - For notebook: open in Jupyter/Colab and run cells for interactive plots.
4. Produced CSV exports (company_monthly_summary.csv, product_summary.csv, cohort_retention.csv, cleaned_online_retail.csv, rfm_table.csv) will be in the repository root for Tableau or further analysis.


