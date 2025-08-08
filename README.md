# ⚙ Predictive Maintenance — A Failure Analysis Approach  
**Identifying Operational Thresholds for Predictive Maintenance**  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)  
![Pandas](https://img.shields.io/badge/Library-Pandas-orange.svg)  
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-yellow.svg)  
![Seaborn](https://img.shields.io/badge/Library-Seaborn-lightblue.svg)  
![SQL](https://img.shields.io/badge/Queries-SQL-green.svg)  
![Dataset](https://img.shields.io/badge/Data-Kaggle-blueviolet.svg)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)  

> ⚡ **Why wait for machines to fail when you can see it coming?**  
> This project uses **failure analysis** to spot early warning signs of mechanical failure in a milling process — helping prevent downtime, extend machine life, and cut costs.  

---

## 📌 Overview  
This project demonstrates how **data-driven analysis** can reduce unplanned downtime by **detecting operational thresholds** that signal machine failure risks.  
Using a synthetic dataset simulating a **milling process**, we performed **Exploratory Data Analysis (EDA)**, correlation studies, and threshold identification to support predictive maintenance strategies.

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
- **Note**: This is **synthetic data**, allowing experimentation without risk to actual equipment.

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
   - Boxplots for thresholds  
   - Histograms for distributions  
   - Heatmaps for correlations  
   - Pairplots for interactions  

---

## 📊 Key Findings  
- **Tool Wear > 200 min** greatly increases failure likelihood.  
- **Torque > 60 Nm** combined with high tool wear is a strong risk factor.  
- **Process Temperature > 305 K** often coincides with failures.  
- Strong correlation between air and process temperature affects operational stress.  

---

## 💻 Reproduction Guide  
**Requirements**:  
`pandas`, `matplotlib`, `seaborn`, `numpy`, `sqlite3`  

**Run the notebook**:  
```bash
git clone https://github.com/<your-username>/predictive-maintenance-failure-analysis.git
cd predictive-maintenance-failure-analysis
jupyter notebook notebooks/predictive_maintenance_analysis.ipynb
