# Diabetes Prediction Web App

This project is a **Machine Learning-based web application** that predicts whether a person is diabetic or not based on various medical features. It is built using **Python**, trained on the **PIMA Indian Diabetes Dataset**, and deployed using **Streamlit** for a simple user interface.

## Files
- `diabetes_model_training.ipynb`: Model training on Google Colab
- `SVM_trained_model.sav`: Trained SVM model
- `STD_trained_model.sav`: StandardScaler used for normalization
- `predictive_system.py`: Local prediction script
- `diabetes_prediction_web_app.py`: Streamlit web app

## Features

- Used machine learning models like **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)**.
- Performed **data preprocessing** and standardization using `StandardScaler`.
- Visualized the dataset using **Seaborn** and **Matplotlib** (boxplots, heatmaps, etc.)
- Applied **hyperparameter tuning** for KNN by testing various `k` values and analyzing decision boundaries.
- Achieved:
  - **SVM Accuracy**: 77.2% on test data  
  - **KNN Accuracy**: 78.8% on test data
- Evaluated model performance using:
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-score, etc.)
- Deployed the final app using **Streamlit** for real-time prediction input.

---

## Tools and Libraries

- Python
- NumPy
- Pandas
- Scikit-learn (SVM, KNN, LR accuracy_score, train_test_split, etc.)
- Seaborn & Matplotlib (data visualization)
- Streamlit (for web deployment)

---

## Dataset

- **PIMA Indian Diabetes Dataset**  
  Contains features such as Glucose, Blood Pressure, BMI, Age, Insulin, and more for diabetes prediction.

---
