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
- **Target Classes:** 2 (Benign = 0, Malignant = 1)
- **Missing Values:** None

---

## 🛠 Technologies Used
- **Language:** Python 3.x
- **Libraries:**
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib, Seaborn (for visualization)
  - Joblib (for model saving)

---

## 📋 Project Workflow
1. **Data Collection:** Load dataset from `sklearn.datasets`
2. **Data Preprocessing:** Normalize, handle missing values (if any)
3. **Exploratory Data Analysis (EDA):** Visualizations and insights
4. **Feature Selection:** Select important features using correlation
5. **Model Training:** Train classifiers like Logistic Regression, SVM, Random Forest, etc.
6. **Model Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC
7. **Model Deployment:** Save the trained model using Joblib

---

## ⚙ Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Steps

```bash
# Clone the repository
git clone https://github.com/satheeshMulinti/Breast-cancer-classification-using-ML-Techniques.git

# Navigate to the project directory
cd Breast-cancer-classification-using-ML-Techniques

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train and Evaluate the Model

```bash
python breast_cancer_classification.py
```

### Jupyter Notebook

```bash
jupyter notebook breast_cancer_analysis.ipynb
```

---

## 🏃 Model Training

The project uses multiple models:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

Example code:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
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
```

---

## 📏 Evaluation

We evaluate using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **ROC-AUC Score**

Example:

```python
from sklearn.metrics import confusion_matrix, roc_auc_score

print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
```

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 96.5% |
| Support Vector Machine | 97.4% |
| **Random Forest Classifier** | **98.2%** |
| K-Nearest Neighbors | 95.6% |
| Decision Tree | 94.7% |
| Naive Bayes | 94.0% |

**Best Model:** Random Forest Classifier  
**Accuracy:** ~98%

**Key Features Influencing Classification:**
- Mean Radius
- Mean Perimeter
- Mean Concave Points
- Mean Texture

---

## 🚀 Future Enhancements
- Implement Deep Learning models (ANN/CNN).
- Create a web interface using Streamlit or Flask.
- Deploy the model on cloud platforms (AWS, Heroku).
- Add real-time prediction from new patient data.
- Integrate with medical record systems for automated screening.

---

## 🤝 Contributing

Contributions are welcome! If you want to improve this project:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For any queries or suggestions, feel free to reach out:
- **GitHub:** [satheeshMulinti](https://github.com/satheeshMulinti)

---

⭐ **If you find this project helpful, please give it a star!** ⭐
