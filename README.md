# Machine Learning Lab

A structured collection of weekly laboratory experiments, Jupyter Notebooks, datasets, and lab reports for the **Machine Learning Lab** course.

---

## 📁 Repository Structure

```text
├── code/       # Jupyter notebooks (.ipynb) for lab experiments & course project
├── pdf/        # Lab reports, problem statements, and course handouts (.pdf)
└── dataset/    # Datasets used across experiments (.csv)
```

---

## 🧪 Experiments Overview

| Lab | Topic | Code | Report | Dataset |
|:---:|:---|:---:|:---:|:---|
| **01** | Data Preprocessing, Imputation & Normalization | — | [Lab1.pdf](pdf/Lab1.pdf) | `synthetic_dataset.csv` |
| **02** | Handling Missing Values & Label Encoding | [Lab2.ipynb](code/Lab2.ipynb) | [Lab2.pdf](pdf/Lab2.pdf) | `synthetic_dataset.csv` |
| **03** | Exploratory Data Analysis (EDA) & Visualization | [Lab3.ipynb](code/Lab3.ipynb) | [Lab3.pdf](pdf/Lab3.pdf) | `df_final_features.csv` |
| **04** | Simple & Multiple Linear Regression | [Lab4.ipynb](code/Lab4.ipynb) | [Lab4.pdf](pdf/Lab4.pdf) | `df_final_features.csv` |
| **05** | Random Forest Classification & Evaluation Metrics | [Lab5.ipynb](code/Lab5.ipynb) | [Lab5.pdf](pdf/Lab5.pdf) | `loan_approval_dataset.csv` |
| **06** | Unsupervised Clustering (K-Means & DBSCAN) | [Lab6.ipynb](code/Lab6.ipynb) | [Lab6.pdf](pdf/Lab6.pdf) | `Mall_Customers.csv` |
| **07** | Binary Classification with Logistic Regression | [Lab7.ipynb](code/Lab7.ipynb) | [Lab7.pdf](pdf/Lab7.pdf) | `loan_approval_dataset.csv` |
| **08** | Decision Tree Classifier & Depth Comparison | [Lab8.ipynb](code/Lab8.ipynb) | [Lab8.pdf](pdf/Lab8.pdf) | `loan_approval_dataset.csv` |

---

## 📌 Course Project: Workspace Rental Price Tier Prediction

* **Notebook:** [`code/classification-algorithms.ipynb`](code/classification-algorithms.ipynb)
* **Web App:** [`app.py`](app.py) (Streamlit interactive prediction & model benchmark dashboard)
* **Overview:** Multi-class classification project evaluating 8 machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, Soft Voting, Stacking Ensemble) to predict workspace rental pricing tiers (*Budget, Standard, Premium, Luxury*).

### 🖥️ Run the Streamlit Application
```bash
streamlit run app.py
```
*(Or double-click `run_app.bat` on Windows)*

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Core Libraries:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Scikit-Learn`
* **Tools:** Jupyter Notebook, Google Colab

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Aruveer/Machine-Learning-Lab.git
cd Machine-Learning-Lab
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebooks
```bash
jupyter notebook code/
```
