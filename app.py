import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

with open(MODEL_DIR / "rdf.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open(MODEL_DIR / "scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        values = [float(request.form[column]) for column in FEATURE_COLUMNS]
    except (KeyError, TypeError, ValueError):
        return render_template(
            "predict.html",
            error="Please complete every field with valid numeric values.",
        ), 400

    applicant_data = pd.DataFrame(
        np.array(values).reshape(1, -1),
        columns=FEATURE_COLUMNS,
    )
    scaled_values = scaler.transform(applicant_data)
    scaled_values = pd.DataFrame(scaled_values, columns=FEATURE_COLUMNS)

    prediction = model.predict(scaled_values)
    approved = int(prediction[0]) == 1

    result = "Loan Approved" if approved else "Loan Rejected"
    status = "approved" if approved else "rejected"

    return render_template("submit.html", prediction=result, status=status)


if __name__ == "__main__":
    app.run(debug=True)
