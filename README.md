# 🎗 Breast Cancer Classification Using Machine Learning Techniques

## 📌 Overview
This project aims to classify whether a tumor is **benign** or **malignant** based on medical diagnostic measurements.  
Using **Machine Learning techniques**, we train models on publicly available medical datasets to help in early detection and decision-making, which can potentially save lives.

---

## 🗂 Table of Contents
1. [Project Motivation](#-project-motivation)
2. [Dataset](#-dataset)
3. [Technologies Used](#-technologies-used)
4. [Project Workflow](#-project-workflow)
5. [Installation](#-installation)
6. [How to Run](#-how-to-run)
7. [Model Training](#-model-training)
8. [Evaluation](#-evaluation)
9. [Results](#-results)
10. [Future Enhancements](#-future-enhancements)
11. [Contributing](#-contributing)
12. [License](#-license)

---

## 🎯 Project Motivation
Breast cancer is one of the most common cancers among women worldwide.  
Early detection plays a crucial role in successful treatment.  
By leveraging **Machine Learning**, we can classify cancerous tumors based on diagnostic features and assist medical professionals in their decisions.

---

## 📊 Dataset
- **Source:** [Breast Cancer Wisconsin Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) or `sklearn.datasets.load_breast_cancer`.
- **Number of Instances:** 569
- **Number of Features:** 30 numeric features (mean radius, mean texture, mean smoothness, etc.)
- **Target Variable:**  
  - `0` → Malignant  
  - `1` → Benign

Example feature list:
- Mean Radius  
- Mean Texture  
- Mean Perimeter  
- Mean Smoothness  
- Mean Compactness  
- ... and more.

---

## 🛠 Technologies Used
- **Programming Language:** Python 3.x
- **Libraries:**
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Model Saving: `joblib` / `pickle`
- **Environment:** Jupyter Notebook / Google Colab

---

## 📋 Project Workflow
1. **Data Collection** – Load the dataset from UCI or sklearn.
2. **Data Preprocessing** – Handle missing values, normalize features, encode target labels.
3. **Exploratory Data Analysis (EDA)** – Visualize feature correlations and class distributions.
4. **Model Training** – Train multiple ML algorithms (Logistic Regression, Random Forest, SVM, KNN, etc.).
5. **Model Evaluation** – Compare metrics to choose the best model.
6. **Model Saving** – Store the trained model for deployment.

---

## ⚙ Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Breast-Cancer-Classification-Using-ML-Techniques.git

# Navigate to the project directory
cd Breast-Cancer-Classification-Using-ML-Techniques

▶ How to Run
Option 1: Run Jupyter Notebook
jupyter notebook Breast_Cancer_Classification.ipynb

Option 2: Run Python Script
python main.py

🏋 Model Training

Example:

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'breast_cancer_model.pkl')

📏 Evaluation

We evaluate using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC Score

Example:

from sklearn.metrics import confusion_matrix, roc_auc_score
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

📈 Results

Best Model: Random Forest Classifier

Accuracy: ~98%

Key Features Influencing Classification:

Mean Radius

Mean Perimeter

Mean Concave Points

Mean Texture

🚀 Future Enhancements

Implement Deep Learning models (ANN/CNN).

Create a web interface using Streamlit or Flask.

Deploy the model on cloud platforms (AWS, Heroku).

Add real-time prediction from new patient data.

# Install dependencies
pip install -r requirements.txt
