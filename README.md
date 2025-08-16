# Framingham CHD Prediction

This project predicts the **10-year risk of Coronary Heart Disease (CHD)** using the **Framingham Heart Study dataset**.

The workflow includes:
- Data exploration & visualization
- Data preprocessing (handling missing values, scaling, SMOTE)
- Model training with multiple algorithms
- Model evaluation using **Accuracy**, **Recall**, **Confusion Matrix**, and **ROC-AUC**

---

##  Dataset
The dataset is based on the **Framingham Heart Study** and contains health-related features such as:
- Age, BMI, Blood Pressure, Glucose levels, etc.
- Target variable: `TenYearCHD` (0 = No CHD, 1 = CHD within 10 years)

> Place the dataset at `data/framingham.csv`.

---

## ⚙ Steps in the Project

1. **Data Loading & Exploration**
   - Display dataset shape, missing values, and target distribution
   - Visualize missing values, correlation heatmap, and feature distributions

2. **Data Preprocessing**
   - Impute missing values using `SimpleImputer` (mean strategy)
   - Standardize features with `StandardScaler`
   - Apply **SMOTE** to balance the dataset

3. **Model Training**
   - Algorithms used:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - XGBoost
     - LightGBM
   - Models evaluated on **accuracy** and **recall for CHD cases**

4. **Model Evaluation**
   - Confusion Matrix heatmap
   - ROC Curve with AUC score
   - Classification Report

---

##  Results (example from one run)

**Best Model:** Logistic Regression (based on recall for CHD)

| Metric           | Value |
|------------------|-------|
| Accuracy         | 0.6557 |
| Recall (CHD=1)   | 0.6016 |
| AUC Score        | ~0.71 |

**Confusion Matrix (example):**
```
[[482, 243],
 [ 49,  74]]
```

**ROC Curve:**
- Shows the trade-off between sensitivity and specificity
- AUC ≈ **0.71**

---

##  Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/Framingham-CHD-Prediction.git
cd Framingham-CHD-Prediction

# (Recommended) Create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add the dataset
# Place your CSV at: data/framingham.csv

# Run the script
python chd_prediction.py
```

---

##  Requirements
See `requirements.txt` for the exact package list:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm

---

##  Notes
- This project focuses on **recall for CHD** to reduce false negatives, as missing CHD cases can be critical.
- Figures are saved to the `outputs/` folder as PNG files after running the script.

---

## 📜 License
This project is licensed under the MIT License.
