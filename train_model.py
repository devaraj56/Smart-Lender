import pickle
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "loan_prediction.csv"
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"

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

CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status",
]


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    print("\n========== Dataset Loaded ==========")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nMissing values before treatment:")
    print(df.isnull().sum())
    return df


def create_eda_charts(df):
    STATIC_DIR.mkdir(exist_ok=True)

    numeric_columns = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    df[numeric_columns].hist(figsize=(12, 4), bins=25)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / "univariate_numeric.png")
    plt.close()

    category_columns = ["Gender", "Education", "Credit_History", "Loan_Status"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, column in zip(axes.ravel(), category_columns):
        df[column].value_counts(dropna=False).plot(kind="bar", ax=axis, color="#0b4f6c")
        axis.set_title(column)
        axis.set_xlabel("")
    plt.tight_layout()
    plt.savefig(STATIC_DIR / "univariate_categorical.png")
    plt.close()

    pd.crosstab(df["Gender"], df["Married"]).plot(kind="bar", figsize=(7, 5))
    plt.title("Gender vs Married")
    plt.tight_layout()
    plt.savefig(STATIC_DIR / "bivariate_gender_married.png")
    plt.close()

    pd.crosstab(df["Education"], df["Self_Employed"]).plot(kind="bar", figsize=(7, 5))
    plt.title("Education vs Self Employed")
    plt.tight_layout()
    plt.savefig(STATIC_DIR / "bivariate_education_employment.png")
    plt.close()


def preprocess_dataset(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
        df["Loan_Amount_Term"].mean()
    )

    for column in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
        df[column] = df[column].fillna(df[column].mode()[0])

    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    encoders = {}
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    print("\nMissing values after treatment:")
    print(df.isnull().sum())
    print("\nLoan status before SMOTE:")
    print(y.value_counts())

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    print("\nLoan status after SMOTE:")
    print(pd.Series(y_balanced).value_counts())

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_balanced),
        columns=FEATURE_COLUMNS,
    )

    return X_scaled, y_balanced, scaler, encoders


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n========== {name} ==========")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
    }


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier()),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ]

    results = [
        evaluate_model(name, model, X_train, X_test, y_train, y_test)
        for name, model in models
    ]

    best = max(results, key=lambda result: result["accuracy"])
    cv_scores = cross_val_score(best["model"], X, y, cv=5)

    print("\n========== Best Model ==========")
    print("Model:", best["name"])
    print("Test Accuracy:", best["accuracy"])
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation accuracy:", cv_scores.mean())

    return best, cv_scores


def save_artifacts(best, scaler, encoders, cv_scores):
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = save_pickle(best["model"], MODEL_DIR / "rdf.pkl")
    scaler_path = save_pickle(scaler, MODEL_DIR / "scaler.pkl")
    encoders_path = save_pickle(encoders, MODEL_DIR / "encoders.pkl")

    metrics = [
        "Smart Lender Model Metrics",
        f"Best model: {best['name']}",
        f"Test accuracy: {best['accuracy']:.4f}",
        f"Cross-validation scores: {', '.join(f'{score:.4f}' for score in cv_scores)}",
        f"Average cross-validation accuracy: {cv_scores.mean():.4f}",
        f"Model artifact: {model_path.name}",
        f"Scaler artifact: {scaler_path.name}",
        f"Encoders artifact: {encoders_path.name}",
    ]
    (MODEL_DIR / "metrics.txt").write_text("\n".join(metrics), encoding="utf-8")

    print("\nModel, scaler, encoders, and metrics saved successfully.")


def save_pickle(value, path):
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(temp_path, "wb") as artifact_file:
        pickle.dump(value, artifact_file)

    try:
        temp_path.replace(path)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        temp_path.replace(fallback_path)
        print(
            f"Could not replace locked artifact {path.name}. "
            f"Saved refreshed artifact as {fallback_path.name}."
        )
        return fallback_path


def main():
    df = load_dataset()
    create_eda_charts(df)
    X, y, scaler, encoders = preprocess_dataset(df)
    best, cv_scores = train_models(X, y)
    save_artifacts(best, scaler, encoders, cv_scores)


if __name__ == "__main__":
    main()
