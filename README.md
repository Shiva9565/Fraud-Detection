# 💳 Fraud Detection with Machine Learning & Deep Learning

### A robust project that predicts fraudulent financial transactions using both traditional machine learning models and deep learning (ANN).

---

## ✨ Features

- 🔍 **Fraud Detection**: Predicts if a transaction is fraudulent or not.
- 🧠 **ML & DL Models**: Uses algorithms like KNN, GaussianNB, Gradient Boosting, AdaBoost, and a custom Artificial Neural Network.
- 📊 **Data Cleaning & Preprocessing**: Handles missing values, standardizes numeric features, and encodes categorical ones.
- 📈 **Exploratory Data Analysis**: Includes correlation heatmaps, transaction validation, and consistency checks.
- 📌 **Model Evaluation**: Detailed metrics such as Accuracy, Precision, Recall, F1-Score, AUC, Log Loss, Jaccard, MCC, Balanced Accuracy.
- 🧪 **Experiment Tracking**: Compares models on multiple evaluation metrics.
- 📉 **Visualization**: Accuracy bar plots and heatmaps for EDA.

---

## 🛠️ Technologies Used

- **Python**: Core language  
- **TensorFlow/Keras**: For designing and training the ANN  
- **Scikit-learn**: ML models, metrics, preprocessing  
- **Pandas & NumPy**: Data manipulation  
- **Matplotlib & Seaborn**: Visualization  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection-ml-dl.git  
cd fraud-detection-ml-dl  
```

### 2️⃣ Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

### 3️⃣ Required Files

Ensure the following files are present in your project folder:

- `balanced_dataset.csv` – Preprocessed and balanced dataset
- `Fraud.csv` – Original test dataset
- `EDA.ipynb` – Exploratory data analysis
- `databalancing.ipynb` – Data balancing logic
- `experiments.ipynb` – Model training, evaluation & comparison
- `requirements.txt` – Dependencies list

---

## 🧠 Model Development

### ✅ Machine Learning Models
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Gradient Boosting
- AdaBoost

### ✅ Deep Learning Model
- ANN with:
  - 3 hidden layers
  - Batch Normalization and Dropout
  - Binary Crossentropy loss
  - Optimizer: Adam
  - Metrics: Accuracy, Precision, Recall, AUC

---

## 📊 Performance & Visualization

- Bar graph comparing accuracy of all ML and DL models
- Confusion matrix and classification metrics for deep learning
- Correlation heatmap for all numerical features

---

## 🔍 Key Findings

- **Top predictors** of fraud: `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, and their consistency with `newbalanceDest`.
- **Fraudulent patterns**: Transactions where source/destination balance inconsistencies were observed.
- **Model insights**: Gradient Boosting and ANN showed the highest performance in detecting fraud.

---

## 🧠 AI-Driven Insights

The system evaluates patterns in numeric transaction data and flag fraudulent transactions. The pipeline includes feature validation, model comparison, and comprehensive evaluation for production-readiness.

---

## 🧑‍💻 Author

Developed by **Shiva Kant Pandey**

---

## 🛡️ Security & Prevention Suggestions

- Enforce **real-time transaction validation** checks based on balance differences.
- Introduce **thresholding systems** using AUC/F1-score tuned models for live fraud flagging.
- Regular **retraining with updated datasets** to capture evolving fraud tactics.

---

## ✅ Monitoring Effectiveness

To verify if prevention methods work:

- Monitor drop in fraud % post-implementation
- Compare F1-score of historical and new model predictions
- A/B test new infrastructure with a control group

---

Let me know if you'd like this README file exported as `.md` or pushed to a GitHub structure!
