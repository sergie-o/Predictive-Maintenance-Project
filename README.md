# 🚀 Predictive Maintenance – A Failure Analysis Approach  
### 📍 Identifying Operational Thresholds to Predict & Prevent Machine Failures  

**What if we could stop a machine from failing before it even breaks?**  
This project dives into machine performance data to do exactly that — **spot the early warning signs** of mechanical failure and help industrial operators **extend machine life, reduce downtime, and save costs**.  

---

## 📊 Dataset Overview  

- **Source:** [AI4I 2020 Predictive Maintenance Dataset – Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/data)  
- **Type:** Synthetic, but realistic — generated to mimic a **milling process** (when machines cut or shape materials).  
- **Why Synthetic?** To simulate *real-world failure scenarios* without the cost or risk of breaking actual machines.  
- **Size:** 10,000 rows × 14 columns.  
- **Key Variables:**  
  - **Operational:**  
    - `Air temperature [K]`  
    - `Process temperature [K]`  
    - `Torque [Nm]`  
    - `Tool wear [min]`  
    - `Rotational speed [rpm]`  
  - **Target:** `Machine failure` (0 = Success, 1 = Failure)  
  - **Failure Types:** `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), `RNF` (Random Failure)  

---

## 🎯 Project Goal  

**Main Question:**  
💡 *"Can we identify operational thresholds that predict machine failure so maintenance teams can act before breakdowns occur?"*  

The aim was not only to **analyze patterns** but also to define **data-backed intervention points** — the thresholds where a machine is most at risk.  

---

## 🛠 Methodology – What I Did  

1. **Data Cleaning** – Formatted data, removed inconsistencies, ensured correct data types.  
2. **Exploratory Data Analysis (EDA)** –  
   - Heatmaps to reveal correlations.  
   - Boxplots & histograms to compare success vs. failure distributions.  
   - Threshold detection for operational variables.  
3. **SQL Analysis** – Wrote queries to answer key business questions directly from the dataset.  
4. **Visualization** – Python (`matplotlib`, `seaborn`) & Tableau for clear, actionable visuals.  

---

## 📌 Key Insights – The “Aha!” Moments  

- **Tool Wear:** Failures spike sharply after **175 minutes** of tool wear.  
- **Process Temperature:** Operating above **310 K** significantly raises failure risk.  
- **Torque:** Readings above **60 Nm** are a common precursor to failures.  
- **Failure Modes:** Certain product types are disproportionately linked to specific failures.  

These insights form **data-driven maintenance rules** — thresholds that can be monitored in real time to avoid costly downtime.  

---

## 💻 How to Reproduce This Project  

**Requirements:**  
- Python **3.10+**  
- SQLite3 installed  
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter  

**Steps:**  

```bash
# 1️⃣ Clone this repository
git clone https://github.com/yourusername/predictive-maintenance-failure-analysis.git
cd predictive-maintenance-failure-analysis

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ (Optional) Run SQL queries for analysis
sqlite3 data/cleaned_data.db < sql/predictive_maintenance_queries.sql

# 4️⃣ Open the Jupyter Notebook for EDA & visualizations
jupyter notebook notebooks/predictive_maintenance_analysis.ipynb
