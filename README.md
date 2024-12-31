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

### 3. Modeling, Tracking, and Versioning

In the [`modeling.ipynb`](./modeling.ipynb) notebook, we experimented with a range of machine learning models, including:

- Random Forest  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  

Recognizing the imbalanced nature of our target, we incorporated boosting algorithms such as XGBoost and LightGBM. These were trained on both SMOTE-balanced and imbalanced data, leveraging their strong performance on skewed datasets. Model performance metrics and artifacts were tracked using MLflow, which was synced with our Dagshub repository. Additionally, data versioning was managed through DVC to maintain control over dataset iterations.

### 4. Front End and Back End

The back end is powered by FastAPI for its lightweight nature and efficient Docker compatibility. The front end, developed in Streamlit, provides a straightforward interface to interact with and test the deployed model. Both components are containerized using Dockerfiles and dedicated requirements files to streamline the build and deployment processes.

### 5. Deployment

Deployment to Microsoft Azure leverages the Azure Container Registry (ACR) for storing both the back end and front end Docker images. Azure Container Instances (ACI) then hosts and runs these containerized services, ensuring a scalable and flexible environment for the entire application.

### 6. CI/CD Pipeline

A Continuous Integration/Continuous Deployment (CI/CD) pipeline built with GitHub Actions automates the workflow. Key steps include:

1. Checking out the repository code  
2. Installing and configuring the Azure CLI  
3. Logging in to Azure and the ACR  
4. Building and pushing the back end and front end images  
5. Generating the `container-group.json` file  
6. Deploying the application to Azure Container Instances (ACI)

This pipeline ensures reliability, reproducibility, and efficient delivery of updates throughout the machine learning lifecycle.

## Resources

- **Dataset:** [Insurance Dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data)

