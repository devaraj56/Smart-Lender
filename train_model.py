from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
df = pd.read_csv("dataset/loan_prediction.csv")
print(df.head())

print("Shape of Dataset:")
print(df.shape)

print(df.info())

print(df.isnull().sum())
print(df.describe())
print(df.select_dtypes(include="object").columns)
# Handling missing values

# Numerical columns - fill with mean
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())

# Categorical columns - fill with mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

print("\nMissing Values After Treatment:")
print(df.isnull().sum())
df.drop("Loan_ID", axis=1, inplace=True)

print("\nColumns after removing Loan_ID:")
print(df.columns)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_columns = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area',
    'Loan_Status'
]

for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

print("\nEncoded Dataset:")
print(df.head())
print("\nUpdated Data Types:")
print(df.dtypes)
# Split features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nLoan Status Before SMOTE:")
print(y.value_counts())
smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X, y)

print("\nAfter SMOTE")
print(pd.Series(y_smote).value_counts())
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_smote)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Data:")
print(X_scaled.head())
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_smote,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
print("\n========== Decision Tree ==========")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Accuracy :", accuracy_score(y_test, dt_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report")
print(classification_report(y_test, dt_pred))
print("\n========== Random Forest ==========")

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))
print("\n========== KNN ==========")

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("Accuracy :", accuracy_score(y_test, knn_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, knn_pred))

print("\nClassification Report")
print(classification_report(y_test, knn_pred))
print("\n========== Gradient Boosting ==========")

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)
print("\n========== Cross Validation ==========")

cv_scores = cross_val_score(rf, X_scaled, y_smote, cv=5)

print("Cross Validation Scores:")
print(cv_scores)

print("Average Accuracy:", cv_scores.mean())
print("Accuracy :", accuracy_score(y_test, gb_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, gb_pred))

print("\nClassification Report")
print(classification_report(y_test, gb_pred))
pickle.dump(rf, open("model/rdf.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("\nModel and Scaler saved successfully!")