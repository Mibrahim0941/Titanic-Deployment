# Titanic Survival Prediction & Deployment

This repository contains the code for predicting Titanic passenger survival using a Random Forest model, along with its deployment configuration on Azure using GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD).

## Project Overview

### What I Did
- **Machine Learning Model**: Built and trained a classification model to predict the survival of passengers on the Titanic.
- **Cloud Deployment**: Deployed the trained model and its web application interface to Microsoft Azure.
- **CI/CD Pipeline**: Set up an automated deployment pipeline connecting GitHub and Azure.

### How I Did It
1. **Model Training**: 
   - Used the well-known **Titanic dataset**.
   - Performed data preprocessing (handling missing values, encoding categorical features).
   - Trained a **Random Forest Classifier** to predict the target variable (survival).
   - Saved the trained model and encoders as pickle files for inference.
2. **Azure Deployment**: 
   - Created a Web App instance on Microsoft Azure to host the application.
3. **CI/CD with GitHub Actions**:
   - Connected the GitHub repository to the Azure Web App.
   - Autoconfigured **GitHub Actions** to handle the deployment workflow.
   - The workflow is triggered automatically whenever changes to the backend code are pushed to the main Git branch. This ensures that the latest version of the app is continuously integrated, tested, and redeployed to Azure seamlessly.

## Repository Structure
- `app.py`: The main web application script (e.g., Flask/FastAPI) that serves the model predictions.
- `train.py`: The script used for data preprocessing and training the Random Forest model.
- `model.pkl`: The serialized trained Random Forest model.
- `*.pkl`: Various serialized encoders and statistics (e.g., age medians, embarked encoder, etc.) used for preprocessing user input.
- `.github/workflows/`: Contains the GitHub Actions workflow YAML files for CI/CD.
- `requirements.txt`: Python dependencies required to run the application.
- `templates/` & `static/`: HTML templates and static assets for the web interface.
