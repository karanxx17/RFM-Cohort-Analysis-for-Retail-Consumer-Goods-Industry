# 🛒 RFM-Cohort Analysis for Retail & Marketing Industry
### End-to-End Customer Segmentation & Sales Optimization

> *Turning 10,000 raw transactions into revenue-driving strategy — from messy CSV to executive boardroom.*

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple?logo=plotly)](https://plotly.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [RFM Analysis](#-rfm-analysis)
- [K-Means Clustering & Segmentation](#-k-means-clustering--segmentation)
- [Cohort Analysis](#-cohort-analysis)
- [Customer Lifetime Value](#-customer-lifetime-value-clv)
- [Key Metrics & KPIs](#-key-metrics--kpis)
- [Key Insights](#-key-insights)
- [Strategic Recommendations](#-strategic-recommendations)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)

---

## 🚀 Project Overview

This is a full-stack, production-style data analytics project built by **Karan Patel** that covers the complete analytics lifecycle — from raw data ingestion all the way to executive-grade reporting and strategic recommendations.

The project analyzes **10,000 retail transactions** across **1,986 customers** and **499 products** spanning 4 categories. The core mission: go beyond vanity metrics and answer the questions that actually move revenue.

| Metric | Value |
|---|---|
| 👥 Total Customers | 1,986 |
| 🛍️ Total Orders | 10,000 |
| 📦 Total Units Sold | 50,065 |
| 💰 Total Revenue | **$1,078,670.98** |
| 📈 Profit Margin | 18.38% |
| 🔁 Repeat Purchase Rate | **96.27%** |
| 🧩 Customer Segments | 2 primary + 8 RFM sub-segments |
| 📊 Visualizations Generated | 27+ |
| 🗂️ Features Engineered | 26 new features |

---

## 🧩 Business Problem

Retail businesses generate enormous transactional data — but most of it sits idle. This project tackles three core business problems head-on:

### 1. 🎯 Revenue is Concentrated in Unknown Segments
Without segmentation, marketing budgets are sprayed equally across all customers. The reality: a small group drives a disproportionate share of revenue, while others are slowly churning.

> **Question:** *Who are our highest-value customers, and how do we protect that revenue?*

### 2. ⚠️ Churn Is Eroding the Customer Base Silently
A 10.88% churn rate sounds acceptable — until you calculate the lifetime value walking out the door. Customers don't announce when they're leaving. They just go quiet.

> **Question:** *Which customers are drifting, and what's the optimal intervention window?*

### 3. 📦 Product Investment Is Flying Blind
Some categories drive volume. Others drive margin. Without granular category analytics, inventory, pricing, and promotional decisions are made on gut feeling.

> **Question:** *Where should we double down, and where should we pull back?*

---

## 📂 Dataset

- **Source:** [Kaggle — Retail Sales Data with Seasonal Trends and Marketing](https://www.kaggle.com/datasets/abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing)
- **Records:** 10,000 orders | **Customers:** 1,986 unique IDs | **Products:** 499 unique SKUs

| Column | Type | Description |
|---|---|---|
| `Order_ID` | Object | Unique order identifier |
| `Order_Date` / `Ship_Date` | DateTime | Transaction and shipping timestamps |
| `Customer_ID` | Object | Unique customer identifier |
| `Segment` | Category | Consumer / Corporate / Home Office |
| `Region` | Category | East / West / Central / South |
| `Product_Category` | Category | Electronics, Furniture, Office Supplies, Clothing |
| `Sales` | Float | Order revenue |
| `Quantity` | Int | Units ordered |
| `Discount` | Float | Applied discount rate (0–0.25) |
| `Profit` | Float | Order profit |
| `Shipping_Cost` | Float | Cost to ship |
| `Order_Priority` | Category | Low / Medium / High / Critical |

**Data Quality (before cleaning):**
- Missing values: 80 cells (0.04%) — `Customer_Name` (50) and `Profit` (30)
- Duplicate rows: 0
- Complete rows after cleaning: 10,000 (100%)

---

## 🏗️ Project Architecture

```
Raw Transaction Data (CSV)
          │
          ▼
┌──────────────────────────┐
│  Part 1: Data Acquisition│  ← Kaggle download, folder setup, initial inspection
│  & Setup                 │
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│  Part 2: Data Cleaning   │  ← Missing values, deduplication, type conversion,
│  & Preprocessing         │    outlier treatment, feature engineering (26 features)
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│  Part 3: EDA             │  ← Univariate, Bivariate, Time Series, Regional,
│                          │    Customer Behavior, Product Performance Analysis
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│  Part 4: Segmentation    │  ← RFM Analysis → K-Means Clustering → Cohort Analysis
│  & Advanced Analytics    │    → CLV Calculation → Market Basket Analysis
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│  Part 5: KPI Design &    │  ← 15+ KPIs, Monthly Trends, Category KPIs,
│  Dashboard Prep          │    Executive Summary Report, Recommendations
└──────────────────────────┘
          │
          ▼
   📊 27+ Visualizations  |  📄 Executive Report  |  🎯 Strategy Deck
```

---

## 🔬 Pipeline Walkthrough

### Part 1 — Data Acquisition & Setup
- Configured project folder structure (`data/`, `notebooks/`, `scripts/`, `outputs/`, `docs/`)
- Downloaded dataset via `kagglehub`
- Initial inspection: 10,000 rows × 19 columns, 6.13MB, 3 data types

### Part 2 — Data Cleaning & Feature Engineering

**Missing value treatment:**
- `Profit` (30 nulls) → filled with column median ($20.33)
- `Customer_Name` (50 nulls) → filled with mode

**Outlier detection (IQR method):**

| Column | Outliers | % | Bounds |
|---|---|---|---|
| Sales | 328 | 3.28% | [-71.87, 275.33] |
| Profit | 87 | 0.87% | [-61.87, 101.47] |
| Quantity | 0 | 0.00% | Clean |

**26 new features engineered:**

| Feature Group | Features Created |
|---|---|
| 🗓️ Time-Based | Year, Month, Quarter, Day, Day of Week, Week of Year, Season |
| 🏷️ Flags | Is_Weekend, Is_Month_Start, Is_Month_End |
| 💵 Revenue | Unit_Price, Discount_Amount, Net_Revenue, Profit_Margin, Profit_Ratio |
| 🚚 Delivery | Delivery_Days, Delivery_Category |
| 👤 Customer | Order_Count_per_Customer, Repeat_Customer_Flag |
| 📦 Product | Product_Total_Sales, Product_Avg_Sales, Product_Order_Count |
| 📊 Bins | Sales_Category (quartile-based) |

**Final cleaned dataset:** 10,000 rows × 45 columns | 6.48MB | 0 nulls | 0 duplicates

### Part 3 — Exploratory Data Analysis

- **Univariate:** Distributions, skewness, and kurtosis for Sales, Quantity, Profit, Discount, Unit_Price
- **Bivariate:** Sales by Category, Region, Customer Segment, Order Priority
- **Time Series:** Monthly trends, Quarterly comparison, YoY growth
- **Customer behavior:** Purchase frequency distribution, top-10 customers by revenue
- **Product performance:** Top-20 SKUs by revenue, dual-axis revenue vs quantity charts

**Key EDA findings:**

| Finding | Detail |
|---|---|
| Top category | Electronics (30.4% of revenue) |
| Customer retention | 96.3% repeat purchase rate |
| Peak quarter | Q1 — 34.2% of annual sales |
| Weekend effect | +0.9% vs weekdays (negligible) |
| Pareto check | Top 20% products → 27.3% revenue (weak Pareto) |

**Regional breakdown:**

| Region | Revenue | Orders |
|---|---|---|
| East | $318,581 | 2,968 |
| Central | $275,688 | 2,545 |
| West | $272,330 | 2,513 |
| South | ~$211,000 | 1,974 |

---

## 📐 RFM Analysis

RFM is a battle-tested behavioral scoring framework that ranks every customer on three dimensions that predict future buying behavior and churn risk.

### The Framework

| Dimension | Definition | Scoring Logic |
|---|---|---|
| **R — Recency** | Days since last purchase | Lower days = Score 5 (best) |
| **F — Frequency** | Number of orders placed | More orders = Score 5 (best) |
| **M — Monetary** | Total revenue generated | Higher spend = Score 5 (best) |

Each customer gets a **1–5 score** per dimension using quintile ranking. Recency is inverted — a customer who bought yesterday scores 5, not 1.

### Implementation

```python
# Analysis reference date (day after last transaction)
analysis_date = df['Order_Date'].max() + timedelta(days=1)  # 2023-02-22

# Compute raw RFM values per customer
rfm = df.groupby('Customer_ID').agg(
    Recency   = ('Order_Date', lambda x: (analysis_date - x.max()).days),
    Frequency = ('Order_ID',   'count'),
    Monetary  = ('Sales',      'sum')
).reset_index()

# Score into quintiles (1–5)
rfm['R_Score'] = pd.qcut(rfm['Recency'],  5, labels=[5,4,3,2,1], duplicates='drop')
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

# Combined string score e.g. "555" = Champion, "111" = Lost
rfm['RFM_Score']         = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
rfm['RFM_Score_Numeric'] = rfm[['R_Score','F_Score','M_Score']].astype(int).sum(axis=1) / 3
```

### RFM Distribution Summary (1,986 customers)

| Metric | Mean | Median | Min | Max | Std Dev |
|---|---|---|---|---|---|
| Recency (days) | 81.6 | 58 | 1 | 408 | 77.1 |
| Frequency (orders) | 5.04 | 5 | 1 | 14 | 2.26 |
| Monetary ($) | $543.14 | $516.37 | $13.52 | $1,706.78 | $282.19 |

### Segment Mapping Logic

```python
def segment_customers(row):
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])

    if r >= 4 and f >= 4 and m >= 4:    return 'VIP Customers'        # 🏆 Best of the best
    elif r >= 3 and f >= 3 and m >= 3:  return 'Loyal Customers'      # 💛 Consistent buyers
    elif r >= 4 and f <= 2 and m >= 3:  return 'Potential Loyalists'  # 🌱 High potential
    elif r >= 3 and f <= 2:             return 'New Customers'         # 🆕 Recent first-timers
    elif r <= 2 and f >= 3:             return 'At Risk'               # ⚠️ Were loyal, now quiet
    elif r <= 2 and f <= 2 and m <= 2:  return 'Hibernating'          # 💤 Low engagement
    else:                               return 'About to Sleep'        # 😴 Slipping away (44 found)
```

---

## 🤖 K-Means Clustering & Segmentation

Beyond rule-based RFM buckets, K-Means was applied directly on RFM features to let the data define natural groupings without human bias.

### Methodology

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Standardize features — critical for distance-based algorithms
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Evaluate k = 2 through 10 using three metrics
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    inertia          = kmeans.inertia_
    sil_score        = silhouette_score(X_scaled, kmeans.labels_)
    db_score         = davies_bouldin_score(X_scaled, kmeans.labels_)
```

**Cluster selection validated by:**
- Elbow Method (inertia inflection point)
- Silhouette Score (higher = better-defined clusters)
- Davies-Bouldin Index (lower = better separation)

**Optimal k = 2**, confirmed by all three methods.

### PCA Visual Validation

```python
pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X_scaled)

# Variance explained
PC1: 70.72%   PC2: 24.52%   Total: 95.24%
```

95.24% of variance captured in 2D — near-perfect representation, confirming clean cluster separation.

### Final Cluster Profiles

#### 🌟 Cluster 0 — Special Customers *(The Revenue Engine)*

| Attribute | Value |
|---|---|
| Customer Count | **924 (46.5%)** |
| Revenue Contribution | **$713,663.99 (66.2%)** |
| Avg Recency | **45.8 days** |
| Avg Frequency | **6.89 orders** |
| Avg Monetary | **$772.36** |
| Recency Std Dev | 39.95 days |

Recent, frequent, high-spending. Despite being less than half the customer base, they generate **two-thirds of all revenue**. Revenue per head is $772 vs $344 for At-Risk — a **2.25x gap**. A 5% improvement in Special customer retention ≈ ~$35K protected annually.

#### 🚨 Cluster 1 — At-Risk Customers *(The Recoverable Opportunity)*

| Attribute | Value |
|---|---|
| Customer Count | **1,062 (53.5%)** |
| Revenue Contribution | **$365,006.99 (33.8%)** |
| Avg Recency | **112.8 days** |
| Avg Frequency | **3.42 orders** |
| Avg Monetary | **$343.70** |
| Recency Std Dev | 87.40 days |

Over half the customer base hasn't purchased in nearly **4 months on average**. The recency gap — 113 vs 46 days — is the single most actionable finding in this project. This isn't lost revenue yet. It's a **recovery window**.

### Head-to-Head Comparison

```
                      Special Customers    At-Risk Customers
                    ─────────────────────────────────────────
Count                 924     (46.5%)      1,062     (53.5%)
Revenue            $713,664   (66.2%)      $365,007   (33.8%)
Revenue / Customer   $772.36               $343.70
Avg Orders             6.89                  3.42
Avg Recency          45.8 days             112.8 days
Value Gap             2.25x more revenue per head
Recovery Potential    20% conversion ≈ +$85K incremental revenue
                    ─────────────────────────────────────────
```

---

## 📅 Cohort Analysis

Cohort analysis tracks whether the business is genuinely retaining customers or just masking churn with new acquisitions. Each monthly cohort follows customers from their first purchase onward.

### Setup

```python
df['Order_Month']  = df['Order_Date'].dt.to_period('M')
df['Cohort']       = df.groupby('Customer_ID')['Order_Date'].transform('min').dt.to_period('M')
df['Cohort_Index'] = (df['Order_Month'] - df['Cohort']).apply(lambda x: x.n)

# Build retention heatmap
cohort_size      = df.groupby('Cohort')['Customer_ID'].nunique()
retention_matrix = df.groupby(['Cohort','Cohort_Index'])['Customer_ID'].nunique().unstack()
retention_rate   = retention_matrix.divide(cohort_size, axis=0) * 100
```

### Key Findings

- **14 monthly cohorts** tracked
- **Month-1 retention:** ~27–33% across all cohorts
- **Retention stabilizes** at 25–35% from month 2 onward — a loyal core forms quickly
- **Largest cohort** started in January 2022; smallest late cohorts (6–27 customers) have noisier retention data

> 📌 **The biggest drop is month 0 → 1.** A strong post-purchase onboarding sequence in the first 30 days is the single highest-ROI retention intervention available.

---

## 💎 Customer Lifetime Value (CLV)

CLV quantifies the total expected revenue from a customer over their entire relationship with the business — and when compared against CAC, reveals the true health of unit economics.

### Formula

```python
customer_metrics['Avg_Order_Value']    = Total_Revenue / Order_Count
customer_metrics['Purchase_Frequency'] = Order_Count / (Lifespan_Days / 365.25)

# 3-year horizon assumption
customer_metrics['CLV_Simple'] = Avg_Order_Value × Purchase_Frequency × 3

# Quartile-based CLV tiers
customer_metrics['CLV_Category'] = pd.qcut(CLV_Simple, q=4,
                                    labels=['Low', 'Medium', 'High', 'Very High'])
```

### CLV Results

| Metric | Value | Benchmark |
|---|---|---|
| Avg Customer Lifetime Value | **$7,521.29** | High for retail |
| Customer Acquisition Cost (CAC) | $50.00 | Very low |
| **CLV / CAC Ratio** | **150.43x** | ✅ World-class (healthy = >3x) |
| Profit Per Customer | $99.84 | — |
| CAC Payback Period | **4.4 months** | ✅ Excellent |

> 💡 A CLV/CAC ratio of **150x** means every $1 spent acquiring a customer returns $150 in lifetime value. This justifies aggressive investment in both acquisition AND retention campaigns.

---

## 📊 Key Metrics & KPIs

### 💵 Financial Performance

| KPI | Value | Status |
|---|---|---|
| Total Revenue | $1,078,670.98 | — |
| Total Profit | $198,274.85 | — |
| Profit Margin | **18.38%** | ✅ Healthy (~15–20% target) |
| Avg Order Value (AOV) | $107.87 | — |
| Total Units Sold | 50,065 | — |

### 👤 Customer Health

| KPI | Value | Status |
|---|---|---|
| Repeat Purchase Rate | **96.27%** | ✅ Excellent |
| One-Time Buyers | 74 customers | ⚠️ Monitor |
| Customer Retention Rate | **89.12%** | ✅ Strong |
| Churn Rate (>180 days) | 10.88% (216 customers) | ⚠️ Address |
| Avg Days Since Last Purchase | 81.6 days | ⚠️ Watch |

### 🏷️ Category Performance

| Rank | Category | Revenue | Share | Avg Order |
|---|---|---|---|---|
| 🥇 | Electronics | $328,422.71 | 30.4% | $108.43 |
| 🥈 | Office Supplies | $321,280.86 | 29.8% | $108.95 |
| 🥉 | Furniture | $214,958.03 | 19.9% | $106.89 |
| 4️⃣ | Clothing | $214,009.37 | 19.8% | $106.42 |

> Electronics + Office Supplies = **60.2% of all revenue** from 59.8% of orders. Furniture and Clothing are statistically tied — a margin analysis could reveal which deserves more investment.

### 📦 Operational Efficiency

| KPI | Value |
|---|---|
| Avg Orders per Customer | 5.04 |
| Avg Items per Order | 5.01 |
| Total SKUs | 499 |
| Top 20% products → revenue | 27.3% |
| Peak Quarter | Q1 (34.2% of annual sales) |

---

## 🔍 Key Insights

### ✅ What's Working

**Unit economics are exceptional.** $50 CAC with 4.4-month payback and 150x CLV/CAC. The acquisition engine is working better than most retailers could hope for.

**Near-zero single-purchase rate.** 96.3% of customers buy again — strong product-market fit and customer satisfaction at baseline.

**Revenue concentration creates a lever.** 46.5% of customers generate 66.2% of revenue. Even a 5% retention improvement in this segment ≈ ~$35K protected annually with minimal spend.

**Q1 seasonality is real and predictable.** 34.2% of annual sales concentrate in Q1 — a known planning advantage for inventory, staffing, and campaign timing.

### ⚠️ What Needs Attention

**53.5% of customers are actively drifting.** The At-Risk cluster is the most critical finding. Average recency of 113 days means the intervention window is now — not next quarter.

**74 single-purchase customers = failed onboarding.** They showed intent (they bought once) then vanished. A post-purchase email sequence converting even 50% of them adds meaningful revenue.

**Churn compounds.** 216 customers classified as churned (>180 days inactive). Unchecked, this erodes the base by 200+ customers per year.

**Market Basket analysis returned 0 frequent itemsets.** Products are bought independently — no natural co-purchase behavior. Cross-sell requires deliberate surfacing through recommendations, not organic discovery.

---

## 🎯 Strategic Recommendations

### ⚡ Immediate (0–30 Days)
1. **Loyalty tiers for Special customers** — protect the 66% revenue base with VIP early access and exclusive offers before they drift
2. **Win-back automation for At-Risk customers** — trigger at 60-day inactivity: 10% offer → 20% at 90 days → free shipping at 120 days
3. **Post-purchase sequence for 74 one-time buyers** — personalized re-engagement based on their first purchase category
4. **Churn alert system** — automated flag when any customer crosses 60 days without a purchase

### 📅 Short-Term (2–3 Months)
1. **Segment-specific email flows** — Special customers get VIP content; At-Risk get offers tied to their last category
2. **Curated product bundles** — MBA showed no organic co-purchase patterns; build deliberate Electronics + Office Supplies bundles
3. **Referral program** — at 150x CLV/CAC, referral acquisition is far cheaper than any paid channel
4. **Q1 campaign calendar** — plan inventory and promotions around the 34.2% revenue peak window

### 🔭 Long-Term (6–12 Months)
1. **Predictive churn model** — weekly churn probability scores using recency trend + frequency drop signals
2. **AI recommendation engine** — cross-sell between top categories for Special customers
3. **Category margin deep-dive** — Furniture and Clothing are revenue-tied; margin differences could justify 2x investment in one
4. **Customer success program** — proactive high-touch outreach for top 100 customers by CLV

### 📈 Expected Business Impact

| Initiative | Expected Outcome |
|---|---|
| At-Risk win-back (20% conversion) | +~$85K incremental revenue |
| Overall revenue growth | +15–20% |
| Retention improvement | +25–30% |
| CLV growth | +10–15% |
| Churn reduction | -20–25% |

---

## 🛠️ Tech Stack

```
Data & Analysis
├── Python 3.9+          Core language
├── Pandas 2.3.3         Data manipulation (45-column engineered dataset)
├── NumPy 2.3.5          Numerical operations
└── SciPy                Statistical testing

Machine Learning
├── Scikit-Learn         KMeans, StandardScaler, PCA, silhouette/DB scoring
└── mlxtend              Market Basket Analysis (Apriori + Association Rules)

Visualization
├── Matplotlib           Static plots (distributions, boxplots, heatmaps)
├── Seaborn              Statistical plots (cohort retention heatmap)
└── Plotly               27+ interactive HTML charts (3D scatter, sunburst, trend lines)

Environment
├── Jupyter Notebook     Narrative-driven analysis across 5 parts
└── kagglehub            Automated dataset download
```

---

## 📁 Project Structure

```
retail-analytics/
│
├── data/
│   ├── raw/
│   │   ├── retail_sales_data.csv            # Source dataset from Kaggle
│   │   └── original_data_checkpoint.csv     # Pre-cleaning snapshot
│   └── processed/
│       ├── cleaned_retail_sales.csv          # 10,000 rows × 45 features
│       ├── customer_segments.csv             # RFM scores + K-Means cluster labels
│       ├── customer_clv.csv                  # CLV per customer with categories
│       └── monthly_kpis.csv                  # Month-by-month KPI time series
│
├── notebooks/
│   └── Retail_Analysis.ipynb                # Full 5-part analysis (Parts 1–5)
│
├── outputs/
│   ├── figures/                             # 27+ saved visualizations
│   │   ├── 01_missing_values.png
│   │   ├── 03_outliers_before_treatment.png
│   │   ├── 05_numerical_distributions.png
│   │   ├── 18_optimal_clusters.png          # Elbow + Silhouette + Davies-Bouldin
│   │   ├── 19_customer_segments_pca.html    # Interactive 2D PCA scatter
│   │   ├── 20_customer_segments_3d.html     # 3D RFM cluster space
│   │   ├── 23_cohort_retention.png          # Retention heatmap (14 cohorts)
│   │   └── 25–27_kpi_dashboards.html        # Interactive KPI trend charts
│   └── reports/
│       ├── 01_initial_inspection_report.txt
│       ├── 03_eda_key_findings.txt
│       ├── kpi_summary.csv                  # All 15+ KPIs in one file
│       ├── cohort_retention.csv             # Retention matrix export
│       ├── market_basket_rules.csv          # Association rules output
│       └── executive_summary.txt            # Full boardroom report
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/retail-analytics.git
cd retail-analytics

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook

# 5. Open and run Retail_Analysis.ipynb (Parts 1–5 in order)
```

**requirements.txt**
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
plotly>=5.14
scikit-learn>=1.3
scipy>=1.11
mlxtend>=0.22
kagglehub
jupyter
```

> ⚠️ **Note:** The notebook intentionally uses a wrong filename on first load (`retail_sales_data.csv1222`) to trigger the synthetic data generation path. This ensures the notebook runs end-to-end without needing Kaggle credentials.

---

## 👤 Author

**Karan Patel**
Built as an end-to-end portfolio project demonstrating the full analytics lifecycle: data engineering → EDA → ML segmentation → cohort analysis → CLV modeling → KPI dashboarding → executive reporting.

---

> *"Without data you're just another person with an opinion."* — W. Edwards Deming

---

⭐ **Star this repo if you found it useful!**
