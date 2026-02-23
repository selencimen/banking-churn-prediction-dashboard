# Banking Customer Churn Prediction Dashboard

🚀 End-to-end machine learning pipeline for predicting customer churn using optimized XGBoost with model explainability and interactive Streamlit deployment.

---

## 📌 Project Overview

This project builds a complete machine learning pipeline to predict customer churn in the banking sector.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Baseline Model Comparison
- Hyperparameter Optimization
- XGBoost Final Model
- SHAP Explainability
- Customer Risk Assessment
- Interactive Streamlit Dashboard Deployment

### Final Model Performance
**ROC-AUC: 0.861**

---

### 🖥 Churn Prediction Interface

![Churn Prediction Dashboard](images/9-churn_prediction.png)
## 🧠 Machine Learning Workflow

1. Data preprocessing and encoding  
2. Feature selection and transformation  
3. Baseline models (Logistic Regression, etc.)  
4. Advanced model comparison  
5. Hyperparameter tuning (GridSearch)  
6. Optimized XGBoost model  
7. SHAP-based feature importance analysis  
8. Deployment via Streamlit  

---

## 📊 Interactive Dashboard

The application allows real-time churn probability prediction by entering:

- Credit Score  
- Age  
- Geography  
- Account Balance  
- Estimated Salary  
- Number of Products  
- Tenure  
- Activity Status  

It returns:

- Churn probability  
- Risk classification  
- Model-based recommendations  

---

## 📁 Project Structure

```
banking-churn-prediction-dashboard/
│
├── app.py
├── Churn_Modelling.csv
├── requirements.txt
├── images/
└── README.md
```

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/selencimen/banking-churn-prediction-dashboard.git
cd banking-churn-prediction-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📊 Dataset

Public dataset from Kaggle:  
[Churn Modelling Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)

---

## 🛠 Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- XGBoost  
- SHAP  
- Streamlit  
- Matplotlib  
- Seaborn  

---

## 👥 Project Team

This project was developed as a collaborative group project.

Team Members:
- Selen Cimen
- Laçin Karaarslan
- Dilay Bezazoğlu

---

## 👩‍💻 Author

Selen Cimen  
Data Science & Analytics
