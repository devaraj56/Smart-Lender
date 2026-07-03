from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
model = pickle.load(open("model/rdf.pkl", "rb"))

# Load Scaler
scaler = pickle.load(open("model/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():

    values = [
        float(request.form["Gender"]),
        float(request.form["Married"]),
        float(request.form["Dependents"]),
        float(request.form["Education"]),
        float(request.form["Self_Employed"]),
        float(request.form["ApplicantIncome"]),
        float(request.form["CoapplicantIncome"]),
        float(request.form["LoanAmount"]),
        float(request.form["Loan_Amount_Term"]),
        float(request.form["Credit_History"]),
        float(request.form["Property_Area"])
    ]

    values = np.array(values).reshape(1, -1)

    # Scale the input
    values = scaler.transform(values)

    prediction = model.predict(values)

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template("submit.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)