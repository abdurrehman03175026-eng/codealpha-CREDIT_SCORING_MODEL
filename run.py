from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATASET_PATH = Path(__file__).resolve().parent / "data" / "credit_dataset.csv"
TRAIN_PATH = Path(__file__).resolve().parent / "data" / "train.csv"
TEST_PATH = Path(__file__).resolve().parent / "data" / "test.csv"


def generate_synthetic_credit_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    income = rng.normal(65000, 25000, size=n_samples).clip(12000, 250000)
    age = rng.integers(21, 75, size=n_samples)
    employment_years = rng.integers(0, 40, size=n_samples)
    total_debt = rng.normal(18000, 15000, size=n_samples).clip(0, 120000)
    credit_history_years = rng.integers(0, 30, size=n_samples)
    num_credit_lines = rng.integers(1, 12, size=n_samples)
    num_delinquencies = rng.poisson(0.4, size=n_samples).clip(0, 10)
    payment_history_score = rng.normal(700, 80, size=n_samples).clip(300, 850)
    inquiries_last_6mo = rng.poisson(1.2, size=n_samples).clip(0, 10)

    debt_to_income = total_debt / np.maximum(income, 1)
    credit_utilization = rng.normal(0.34, 0.18, size=n_samples).clip(0.0, 1.0)

    base_score = (
        0.25 * (income / 100000)
        + 0.30 * (payment_history_score / 850)
        - 0.20 * debt_to_income
        - 0.10 * (num_delinquencies / 10)
        - 0.08 * (inquiries_last_6mo / 10)
        + 0.10 * (credit_history_years / 30)
    )
    probability_good = 1 / (1 + np.exp(-6.0 * (base_score - 0.35)))
    is_good_credit = rng.uniform(0, 1, size=n_samples) < probability_good

    data = pd.DataFrame(
        {
            "income": income,
            "age": age,
            "employment_years": employment_years,
            "total_debt": total_debt,
            "credit_history_years": credit_history_years,
            "num_credit_lines": num_credit_lines,
            "num_delinquencies": num_delinquencies,
            "payment_history_score": payment_history_score,
            "inquiries_last_6mo": inquiries_last_6mo,
            "debt_to_income": debt_to_income,
            "credit_utilization": credit_utilization,
            "good_credit": is_good_credit.astype(int),
        }
    )
    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["high_income"] = (df["income"] >= 85000).astype(int)
    df["low_income"] = (df["income"] < 45000).astype(int)
    df["recent_inquiries"] = (df["inquiries_last_6mo"] >= 3).astype(int)
    df["long_credit_history"] = (df["credit_history_years"] >= 10).astype(int)
    df["strong_payment_history"] = (df["payment_history_score"] >= 720).astype(int)
    df["short_employment"] = (df["employment_years"] < 2).astype(int)
    debt_to_income_for_cut = df["debt_to_income"].clip(lower=0.0, upper=1.0)
    df["debt_ratio_category"] = pd.cut(
        debt_to_income_for_cut,
        bins=[-1, 0.2, 0.35, 0.5, 1.0],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)
    return df


def load_or_create_dataset(path: Path = DATASET_PATH, n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    if path.exists():
        print(f"Loading dataset from {path}")
        return pd.read_csv(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating dataset at {path}")
    data = generate_synthetic_credit_data(n_samples=n_samples, random_state=random_state)
    data = engineer_features(data)
    data.to_csv(path, index=False)
    return data


def create_train_test_split(
    data: pd.DataFrame,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_path.exists() and test_path.exists():
        print(f"Loading train data from {train_path}")
        print(f"Loading test data from {test_path}")
        return pd.read_csv(train_path), pd.read_csv(test_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating train data at {train_path}")
    print(f"Creating test data at {test_path}")

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["good_credit"],
    )
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


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
    data = load_or_create_dataset()
    train_df, test_df = create_train_test_split(data)

    feature_columns = [
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

    X_train = train_df[feature_columns]
    y_train = train_df["good_credit"]
    X_test = test_df[feature_columns]
    y_test = test_df["good_credit"]

    logistic_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )
    tree_model = DecisionTreeClassifier(random_state=42, max_depth=7)
    forest_model = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10)

    logistic_pipeline.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    forest_model.fit(X_train, y_train)

    evaluate_model(logistic_pipeline, X_test, y_test, "Logistic Regression")
    evaluate_model(tree_model, X_test, y_test, "Decision Tree")
    evaluate_model(forest_model, X_test, y_test, "Random Forest")

    print("\nFeature importance from Random Forest:")
    importances = pd.Series(forest_model.feature_importances_, index=feature_columns)
    print(importances.sort_values(ascending=False).head(12).to_string())


if __name__ == "__main__":
    main()
