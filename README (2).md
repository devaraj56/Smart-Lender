# Smart Lender: Loan Approval Prediction System

Smart Lender is a machine learning based web application that predicts whether a loan application is likely to be approved or rejected. The project follows a complete ML lifecycle: dataset loading, exploratory data analysis, preprocessing, class balancing, model training, evaluation, artifact saving, and deployment through a Flask web interface.

The application is designed for educational and academic use and demonstrates how predictive analytics can support faster and more consistent loan eligibility decisions.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Workflow](#project-workflow)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Performance](#model-performance)
- [Web Application Flow](#web-application-flow)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Generated Artifacts](#generated-artifacts)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

## Project Overview

Loan approval is traditionally a manual process that requires evaluating applicant income, credit history, employment status, loan amount, property area, and other financial indicators. Manual evaluation can be time-consuming and inconsistent.

Smart Lender uses supervised machine learning to automate this evaluation process. A trained classification model receives applicant details from a Flask web form and returns a prediction:

- Loan Approved
- Loan Rejected

The project uses the loan prediction dataset and compares multiple classification algorithms before saving the best-performing model for deployment.

## Problem Statement

The goal of this project is to build a machine learning system that can predict loan approval status based on applicant information. The system should:

- Analyze historical loan applicant data.
- Handle missing values and categorical features.
- Balance the target classes to reduce biased predictions.
- Train and compare multiple classification models.
- Save the best model for deployment.
- Provide a simple web interface for real-time predictions.

## Project Workflow

The project is implemented according to the workflow described in the Smart Lender project document.

### Epic 1: Data Collection and Architecture Design

- Dataset is stored in `dataset/loan_prediction.csv`.
- Application architecture includes:
  - Dataset layer
  - Training and preprocessing pipeline
  - Saved model artifacts
  - Flask backend
  - HTML templates for user interaction

### Epic 2: Visualizing and Analyzing the Data

The training script performs basic exploratory analysis and generates charts in the `static/` directory:

- Univariate analysis for numerical features.
- Univariate analysis for categorical features.
- Bivariate analysis for `Gender` vs `Married`.
- Bivariate analysis for `Education` vs `Self_Employed`.

Generated chart files:

- `static/univariate_numeric.png`
- `static/univariate_categorical.png`
- `static/bivariate_gender_married.png`
- `static/bivariate_education_employment.png`

### Epic 3: Data Preprocessing

The preprocessing pipeline handles:

- Missing numerical values using mean imputation.
- Missing categorical values using mode imputation.
- Duplicate row removal.
- Removal of non-predictive `Loan_ID`.
- Label encoding for categorical columns.
- SMOTE oversampling to balance loan approval classes.
- StandardScaler normalization for model input features.
- Train-test split for evaluation.

### Epic 4: Model Building

The following models are trained and evaluated:

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors
- Gradient Boosting Classifier

The best model is selected based on test accuracy and validated using 5-fold cross-validation.

### Epic 5: Application Building

The Flask application includes three main pages:

- `home.html`: Landing page with project overview and prediction entry button.
- `predict.html`: Applicant detail form.
- `submit.html`: Prediction result page.

The app loads the saved model and scaler, processes form inputs, scales the features, runs prediction, and displays the result.

## Features

- Real-time loan approval prediction.
- Clean Flask web interface.
- Complete machine learning training pipeline.
- Missing value handling.
- Categorical feature encoding.
- Dataset balancing with SMOTE.
- Feature scaling with StandardScaler.
- Model comparison and evaluation.
- Saved deployment artifacts using Pickle.
- Basic EDA chart generation.
- Input validation in the prediction route.
- Responsive HTML pages.

## Technology Stack

### Language

- Python 3.x

### Backend

- Flask

### Machine Learning and Data Processing

- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Pickle

### Frontend

- HTML5
- CSS3
- Jinja2 templates

## Dataset

The dataset is stored at:

```text
dataset/loan_prediction.csv
```

### Input Features

| Feature | Description |
| --- | --- |
| Gender | Applicant gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Graduate or not graduate |
| Self_Employed | Employment type |
| ApplicantIncome | Applicant income |
| CoapplicantIncome | Co-applicant income |
| LoanAmount | Requested loan amount |
| Loan_Amount_Term | Loan repayment term |
| Credit_History | Credit history status |
| Property_Area | Rural, semiurban, or urban property area |

### Target Variable

| Target | Description |
| --- | --- |
| Loan_Status | Loan approval status, where approved and rejected applications are classified |

## Machine Learning Pipeline

The pipeline is implemented in:

```text
train_model.py
```

Pipeline steps:

1. Load the dataset.
2. Inspect shape, missing values, and target distribution.
3. Generate EDA charts.
4. Fill missing values.
5. Remove unnecessary columns.
6. Encode categorical features.
7. Split features and target.
8. Balance classes using SMOTE.
9. Scale input features using StandardScaler.
10. Split data into training and test sets.
11. Train multiple ML models.
12. Evaluate using accuracy, confusion matrix, and classification report.
13. Apply 5-fold cross-validation.
14. Save model, scaler, encoders, and metrics.

## Model Performance

Latest recorded metrics from `model/metrics.txt`:

| Metric | Value |
| --- | --- |
| Best model | Random Forest |
| Test accuracy | 0.8402 |
| Cross-validation scores | 0.7396, 0.7692, 0.8757, 0.8521, 0.9107 |
| Average cross-validation accuracy | 0.8295 |

The Random Forest model achieved the best performance in the latest training run.

## Web Application Flow

1. User opens the home page at `/`.
2. User clicks `Start Prediction`.
3. User enters applicant details in the prediction form.
4. Flask receives the submitted form data at `/submit`.
5. Input values are converted into the model feature order.
6. Features are scaled using the saved scaler.
7. The trained model predicts approval or rejection.
8. The result page displays the final prediction.

## Project Structure

```text
SmartLender/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- dataset/
|   |-- loan_prediction.csv
|
|-- model/
|   |-- rdf.pkl
|   |-- scaler.pkl
|   |-- encoders.pkl
|   |-- metrics.txt
|   |-- rdf_20260706_174259.pkl
|
|-- static/
|   |-- univariate_numeric.png
|   |-- univariate_categorical.png
|   |-- bivariate_gender_married.png
|   |-- bivariate_education_employment.png
|
|-- templates/
|   |-- home.html
|   |-- predict.html
|   |-- submit.html
```

## Installation and Setup

### 1. Create or activate a Python environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Train the model

Run this command if you want to regenerate model artifacts:

```bash
python train_model.py
```

This creates or updates:

- `model/rdf.pkl`
- `model/scaler.pkl`
- `model/encoders.pkl`
- `model/metrics.txt`
- EDA charts inside `static/`

### Run the Flask application

```bash
python app.py
```

Open this URL in a browser:

```text
http://127.0.0.1:5000/
```

## Generated Artifacts

| File | Purpose |
| --- | --- |
| `model/rdf.pkl` | Deployment model loaded by Flask |
| `model/scaler.pkl` | StandardScaler used to transform form inputs |
| `model/encoders.pkl` | Saved label encoders from training |
| `model/metrics.txt` | Latest model evaluation summary |
| `static/*.png` | EDA visualizations generated by training |

Note: If `rdf.pkl` is locked by Windows or another process during training, the script saves a timestamped model file such as `rdf_20260706_174259.pkl` instead of crashing.

## Example Form Inputs

The prediction form collects:

- Gender
- Married status
- Dependents
- Education
- Self employment status
- Applicant income
- Co-applicant income
- Loan amount
- Loan amount term
- Credit history
- Property area

## Future Enhancements

- Add user authentication.
- Store prediction history in a database.
- Add an admin dashboard for model metrics.
- Deploy the application on a cloud platform.
- Add PDF report generation for each prediction.
- Add more financial indicators for improved accuracy.
- Add automated tests for routes and model prediction behavior.
- Improve model explainability using feature importance or SHAP.

## Learning Outcomes

This project demonstrates:

- End-to-end machine learning workflow.
- Data cleaning and preprocessing.
- Handling class imbalance with SMOTE.
- Model training and comparison.
- Cross-validation based evaluation.
- Saving and loading ML artifacts.
- Flask application development.
- Integrating machine learning with a web interface.

## Author

->Kovuri Udaya Siva Kumar(238x1a4246)
->Gopi Devaraj(238x1a4256)
->Pandaraboina Krishnanjaneyulu (238x1a4281)
->Maddu Nikhil Chandrashekhar (238x1a42f8)
->Thota Praveen Kumar(238x1a42i3)

B.Tech, Computer Science and Engineering  
Artificial Intelligence and Machine Learning

GitHub: [devaraj56](https://github.com/devaraj56/Smart-Lender/)

## License

This project is developed for educational and academic purposes.
