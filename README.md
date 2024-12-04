<<<<<<< HEAD
# MLOps Project: Fraud Detection in Car Accidents Based on Insurance Features

This project, in collaboration with my colleague Med Aziz Ben Lazreg, aims to detect fraudulent car accidents using insurance-related information. The dataset utilized in this project is sourced from Kaggle (see the Resources section below).

## State of the Art

- **Fraud Detection in Automobile Insurance using a Data Mining Based Approach:** [ResearchGate](https://www.researchgate.net/publication/320840047_Fraud_Detection_in_Automobile_Insurance_using_a_Data_Mining_Based_Approach)
- **Fraud Detection in Motor Insurance Claims: Leveraging AI:** [DronaPay Blog](https://www.dronapay.com/post/fraud-detection-in-motor-insurance-claims-leveraging-ai)

## Project Overview

### 1. Exploratory Data Analysis (EDA)

We began with an exploratory data analysis (`fraud_detection_EDA.ipynb`) to extract significant information that would inform the subsequent steps of the project.

### 2. Data Preprocessing

Data preprocessing was performed using pipelines, as implemented in `src/data_preprocessing.py`.

### 3. Modeling

In the modeling phase (`modeling.ipynb`), we tested several models, including:

- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

During the EDA, we found that our target variable was imbalanced. To address this, we employed boosting models such as XGBoost and LightGBM, testing them on both balanced data (using SMOTE) and imbalanced data, due to their strong performance on imbalanced datasets.

## Resources

- **Dataset:** [Insurance Dataset on Kaggle](https://www.kaggle.com/datasets/joudalnsour/insurance/data)
=======
# MLOps project: Fraud detection in car accidents based on insurance-related features
## Ressources:
- Dataset: https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data
>>>>>>> bd934632956c2eb4803fdc3e43b85751a8d5296e
