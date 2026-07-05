from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TEST_PATH = DATA_DIR / "test.csv"

FEATURE_COLUMNS = [
    "income",
    "age",
    "employment_years",
    "total_debt",
    "credit_history_years",
    "num_credit_lines",
    "num_delinquencies",
    "payment_history_score",
    "inquiries_last_6mo",
    "debt_to_income",
    "credit_utilization",
    "high_income",
    "low_income",
    "recent_inquiries",
    "long_credit_history",
    "strong_payment_history",
    "short_employment",
    "debt_ratio_category",
]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> None:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n=== {name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))


def main() -> None:
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["good_credit"]

    for model_path in sorted(MODEL_DIR.glob("*.joblib")):
        model_name = model_path.stem.replace("_", " ").title()
        model = joblib.load(model_path)
        evaluate_model(model, X_test, y_test, model_name)


if __name__ == "__main__":
    main()
