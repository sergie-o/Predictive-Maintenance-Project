# ⚙ Predictive Maintenance — A Failure Analysis Approach  
**Identifying Operational Thresholds to Predict & Prevent Machine Failures**  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)  
![Pandas](https://img.shields.io/badge/Library-Pandas-orange.svg)  
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-yellow.svg)  
![Seaborn](https://img.shields.io/badge/Library-Seaborn-lightblue.svg)  
![SQL](https://img.shields.io/badge/Queries-SQL-green.svg)  
![Dataset](https://img.shields.io/badge/Data-Kaggle-blueviolet.svg)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)  

> 🚀 **Predict. Prevent. Prolong.**  
> Unplanned downtime can cost manufacturers **thousands per minute**.  
> This project leverages **data-driven failure analysis** to *spot early warning signs*, define **actionable operational thresholds**, and keep machines running longer, safer, and at lower cost.  

---
---

## 📌 Overview  
This project demonstrates how **data-driven analysis** can reduce unplanned downtime by detecting **operational thresholds** that signal machine failure risks.  
Using a synthetic dataset simulating a **milling process**, I performed **Exploratory Data Analysis (EDA)**, correlation studies, and threshold identification to support predictive maintenance strategies.

---

## 📂 Dataset Description  
- **Source**: [Predictive Maintenance Dataset (AI4I 2020)](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/data)  
- **Size**: 10,000 rows × multiple operational and failure-related variables  
- **Key Variables**:  
  - `Air temperature [K]`  
  - `Process temperature [K]`  
  - `Torque [Nm]`  
  - `Tool wear [min]`  
  - `Rotational speed [rpm]`  
  - `Machine failure` (binary)  
  - Failure types: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`  
- **Note**: This is **synthetic data**, generated using simulation models to mimic real-world machine behavior.

---

## 🎯 Research Goal  
> **Main Question:**  
> How can operational thresholds be detected to predict and prevent machine failures effectively?

---

## 🛠 Steps Taken  
1. **Data Cleaning**
   - Removed anomalies, renamed columns, encoded categories.  
2. **EDA**
   - Explored variable distributions, relationships, and failure patterns.  
3. **Visualization**
   - Boxplots for threshold detection  
   - Histograms for distribution analysis  
   - Heatmaps for correlation checks  
   - Pairplots for variable interactions  
4. **SQL Analysis**
   - Wrote queries to extract failure counts, success rates, and variable relationships directly from the database.  

---

## 📊 Key Findings  
- **Tool Wear > 175 min** significantly increases the probability of machine failure.  
- **Torque > 60 Nm** combined with high tool wear is a strong risk factor.  
- **Process Temperature > 310 K** often coincides with failures.  
- Air temperature and process temperature are highly correlated, influencing operational stress levels.  

---

## 💻 Reproduction Guide  
**Requirements**:  
`pandas`, `matplotlib`, `seaborn`, `numpy`, `sqlite3`  

**Run the notebook**:  
```bash
git clone https://github.com/<your-username>/predictive-maintenance-failure-analysis.git
cd predictive-maintenance-failure-analysis
jupyter notebook notebooks/predictive_maintenance_analysis.ipynb


## 📁 Repository Structure  
predictive-maintenance-failure-analysis/
│
├── data/                               # Raw and cleaned datasets
│   ├── cleaned_data.db                  # SQLite database with processed data
│   └── raw_data.csv                     # Original dataset
│
├── notebooks/                          # Jupyter notebooks for analysis
│   └── predictive_maintenance_analysis.ipynb
│
├── sql/                                # SQL queries for analysis
│   └── predictive_maintenance_queries.sql
│
├── visuals/                            # Generated plots and charts
│
├── README.md                           # Project documentation
└── requirements.txt                    # Dependencies list
