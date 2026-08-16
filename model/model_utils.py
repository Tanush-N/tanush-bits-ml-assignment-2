from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from model.constants import CLASS_NAMES, FEATURE_COLUMNS, POSITIVE_LABEL, TARGET_COLUMN
except ModuleNotFoundError:
    from constants import CLASS_NAMES, FEATURE_COLUMNS, POSITIVE_LABEL, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = PROJECT_ROOT / "training_data.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.csv"


MODEL_CONFIGS = {
    "Logistic Regression": {
        "artifact": PROJECT_ROOT / "model" / "logistic_regression.joblib",
        "estimator": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=5000, random_state=42),
                ),
            ]
        ),
    },
    "Decision Tree": {
        "artifact": PROJECT_ROOT / "model" / "decision_tree.joblib",
        "estimator": lambda: DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
        ),
    },
    "kNN": {
        "artifact": PROJECT_ROOT / "model" / "knn.joblib",
        "estimator": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
    },
    "Naive Bayes": {
        "artifact": PROJECT_ROOT / "model" / "naive_bayes.joblib",
        "estimator": lambda: GaussianNB(),
    },
    "Random Forest": {
        "artifact": PROJECT_ROOT / "model" / "random_forest.joblib",
        "estimator": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
    },
}


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(TRAIN_DATA_PATH)
    return split_features_target(data)


def load_default_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA_PATH)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_columns(data, require_target=True)
    return data[FEATURE_COLUMNS], encode_target(data[TARGET_COLUMN])


def encode_target(target: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        return target.astype(int)

    normalized = target.astype(str).str.strip().str.upper()
    mapping = {"B": 0, "BENIGN": 0, "0": 0, "M": 1, "MALIGNANT": 1, "1": 1}
    encoded = normalized.map(mapping)

    if encoded.isna().any():
        bad_values = sorted(normalized[encoded.isna()].unique())
        raise ValueError(f"Unsupported diagnosis values: {bad_values}")

    return encoded.astype(int)


def validate_columns(data: pd.DataFrame, require_target: bool) -> None:
    required_columns = list(FEATURE_COLUMNS)
    if require_target:
        required_columns.append(TARGET_COLUMN)

    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def train_models() -> dict[str, object]:
    x_train, y_train = load_training_data()
    fitted_models = {}

    for model_name, config in MODEL_CONFIGS.items():
        model = config["estimator"]()
        model.fit(x_train, y_train)
        fitted_models[model_name] = model

    return fitted_models


def load_models() -> dict[str, object]:
    loaded_models = {}
    missing_artifacts = []

    for model_name, config in MODEL_CONFIGS.items():
        artifact_path = config["artifact"]
        if artifact_path.exists():
            loaded_models[model_name] = joblib.load(artifact_path)
        else:
            missing_artifacts.append(model_name)

    if missing_artifacts:
        trained_models = train_models()
        loaded_models.update({name: trained_models[name] for name in missing_artifacts})

    return loaded_models


def predict_scores(model: object, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_data)
        return 1 / (1 + np.exp(-scores))

    return model.predict(x_data)


def evaluate_model(model: object, x_data: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    y_pred = model.predict(x_data)
    y_score = predict_scores(model, x_data)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_all_models(
    models: dict[str, object],
    x_data: pd.DataFrame,
    y_true: pd.Series,
) -> pd.DataFrame:
    rows = []
    for model_name, model in models.items():
        row = {"ML Model Name": model_name}
        row.update(evaluate_model(model, x_data, y_true))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["F1", "AUC", "MCC"],
        ascending=False,
        ignore_index=True,
    )


def prediction_table(model: object, data: pd.DataFrame) -> pd.DataFrame:
    validate_columns(data, require_target=False)
    x_data = data[FEATURE_COLUMNS]
    predictions = model.predict(x_data)
    probabilities = predict_scores(model, x_data)

    result = data.copy()
    result["predicted_diagnosis"] = [CLASS_NAMES[int(value)] for value in predictions]
    result["malignant_probability"] = probabilities
    return result


def model_confusion_matrix(model: object, x_data: pd.DataFrame, y_true: pd.Series) -> np.ndarray:
    y_pred = model.predict(x_data)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
