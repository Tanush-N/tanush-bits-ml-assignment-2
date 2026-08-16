from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report

from model.model_utils import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    METRICS_PATH,
    TARGET_COLUMN,
    evaluate_all_models,
    load_default_test_data,
    load_models,
    model_confusion_matrix,
    prediction_table,
    split_features_target,
    validate_columns,
)


st.set_page_config(
    page_title="Breast Cancer Classification Models",
    layout="wide",
)


@st.cache_resource
def get_models() -> dict[str, object]:
    return load_models()


@st.cache_data
def get_default_data() -> pd.DataFrame:
    return load_default_test_data()


def load_uploaded_or_default() -> pd.DataFrame:
    uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
    if uploaded_file is None:
        return get_default_data()

    return pd.read_csv(uploaded_file)


def format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    return formatted


def plot_confusion_matrix(matrix) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def show_labeled_results(data: pd.DataFrame, selected_model_name: str, selected_model: object, models: dict[str, object]) -> None:
    x_test, y_test = split_features_target(data)
    metrics = evaluate_all_models(models, x_test, y_test)

    st.subheader("Evaluation Metrics")
    st.dataframe(format_metrics(metrics), use_container_width=True, hide_index=True)

    selected_metrics = metrics[metrics["ML Model Name"] == selected_model_name].iloc[0]
    cols = st.columns(6)
    for col, metric_name in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        col.metric(metric_name, f"{selected_metrics[metric_name]:.4f}")

    report = classification_report(
        y_test,
        selected_model.predict(x_test),
        target_names=["Benign", "Malignant"],
        zero_division=0,
        output_dict=True,
    )

    matrix = model_confusion_matrix(selected_model, x_test, y_test)
    chart_col, report_col = st.columns([0.95, 1.05])
    with chart_col:
        st.pyplot(plot_confusion_matrix(matrix), use_container_width=True)
    with report_col:
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


def show_prediction_results(data: pd.DataFrame, selected_model: object) -> None:
    validate_columns(data, require_target=False)
    predictions = prediction_table(selected_model, data)

    st.subheader("Predictions")
    st.dataframe(predictions, use_container_width=True, hide_index=True)

    summary = predictions["predicted_diagnosis"].value_counts().rename_axis("Prediction").reset_index(name="Rows")
    st.subheader("Prediction Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Breast Cancer Classification Models")

    models = get_models()
    selected_model_name = st.sidebar.selectbox("Select model", list(models))
    selected_model = models[selected_model_name]
    data = load_uploaded_or_default()

    st.sidebar.caption(f"Selected: {selected_model_name}")

    st.subheader("Test Data")
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)

    if METRICS_PATH.exists():
        with st.expander("Saved baseline metrics", expanded=False):
            st.dataframe(format_metrics(pd.read_csv(METRICS_PATH)), use_container_width=True, hide_index=True)

    try:
        if TARGET_COLUMN in data.columns:
            show_labeled_results(data, selected_model_name, selected_model, models)
        else:
            show_prediction_results(data, selected_model)
    except ValueError as exc:
        st.error(str(exc))
        st.info(f"Expected feature columns: {', '.join(FEATURE_COLUMNS)}")

    with st.expander("Class Labels", expanded=False):
        st.write({str(label): name for label, name in CLASS_NAMES.items()})


if __name__ == "__main__":
    main()
