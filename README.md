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
> This project applies **failure analysis** to spot early warning signs of mechanical failure in a milling process — helping prevent downtime, extend machine life, and reduce maintenance costs. 
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

## 🔄 Update 1.0 — First Machine Learning Model  

Building on the exploratory and threshold analysis, I applied my **first machine learning model (K-Nearest Neighbors, KNN)** to the predictive maintenance dataset.  

### ⚙️ Approach  
- **Features (X):** Torque, Tool Wear, Rotational Speed, Air Temperature, Process Temperature  
- **Target (y):** Machine Failure (binary: 0 = success, 1 = failure)  
- Applied **feature scaling (StandardScaler)** and split the dataset into training and testing sets.  
- Implemented **baseline KNN (k=5)** for classification.  

### 📊 Results  
- **Class imbalance:** Dataset is **96.6% machine success** vs **3.4% machine failure**.  
- **Accuracy (baseline):** ~96% across k=1–25  
- **Confusion Matrix (k=5):**  
  - True Negatives (TN) = 1914  
  - False Positives (FP) = 11  
  - False Negatives (FN) = 60  
  - True Positives (TP) = 15  
- **Recall (failures):** 0.20 → Only 20% of actual failures detected.  
- **Insight:** High accuracy is misleading because of class imbalance; the model struggles to identify failures.  

### 🧾 Key Takeaways  
- **Recall is more critical than accuracy** in predictive maintenance.  
- False negatives (missed failures) are riskier than false positives (extra checks).  
- Accuracy alone cannot measure model success in imbalanced datasets.  

### 🚀 Next Steps  
- Adjust classification **thresholds** to prioritize recall.  
- Experiment with **distance-weighted KNN**.  
- Compare performance with **Logistic Regression** for better interpretability.  

## 💻 Reproduction Guide  
**Requirements**:  
`pandas`, `matplotlib`, `seaborn`, `numpy`, `sqlite3`  

 **Run Instructions:**
1. **Clone this repository**
   ```bash
   git clone https://github.com/sergie-o/Predictive-Maintenance-Project.git
2. **Navigate to the project folder**
   ```bash
    cd Predictive-Maintenance-Project
3. **Open the Jupyter Notebook**
- If you use Jupyter Notebook:
   ```bash
   jupyter notebook "Project_predictive_maintenance.ipynb"
- Or, open it in VSCode by double-clicking the file or using:
   ```bash
    code "Project_predictive_maintenance.ipynb"
4. **Ensure the dataset is in the correct location**
- The file ai4i2020.csv must be in the same directory as the notebook.
5. Run all cells
- Select Cell > Run All in Jupyter Notebook or VSCode to reproduce the analysis.
## :rocket: Next Steps
- 	Predict Failures Before They Happen – Using the identified operational       thresholds, models can be trained to recognize early warning signs and       flag machines before they reach critical failure points.
-   Keep Machines in the Safe Zone – By continuously monitoring operational      variables (e.g., torque, tool wear, process temperature), predictive         systems can regulate performance to remain within safe operational           ranges, reducing stress on components.
-   Enable Automated Preventive Actions – Integrating these predictions with     control s ystems could automatically trigger adjustments, slowdowns, or      maintenance requests when a threshold is exceeded.

## 📁 Repository Structure  
```bash
predictive-maintenance-failure-analysis/
│
├── data/                               # Raw and cleaned datasets
│   ├── cleaned_data.db                  # SQLite database with processed data
│   └── ai4i2020.csv                     # Original dataset
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
