# ML Assignment 2 - Classification Model Deployment

## a. Problem statement

The objective of this project is to build and deploy an interactive machine learning web application for a binary classification problem. The application trains multiple classification models on the Breast Cancer Wisconsin Diagnostic dataset and evaluates them on held-out test data using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient.

The Streamlit application allows a user to upload a CSV test file, select a model, view evaluation metrics, and inspect the confusion matrix or classification report.

## b. Dataset description

Dataset: Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine Learning Repository.

The dataset contains measurements computed from digitized images of breast mass cell nuclei. Each record is labeled as benign or malignant.

Rows: 569 total instances.

Features: 30 numeric input features.

Target column: `diagnosis`

Target classes:

- `B`: benign
- `M`: malignant

This repository uses an 80:20 stratified split:

- `training_data.csv`: used to train the models
- `test_data.csv`: used for evaluation and Streamlit upload testing
- `dataset.csv`: complete cleaned dataset

## c. Github Repository Link

Add your GitHub repository link here after uploading this project:

`https://github.com/Tanush-N/tanush-bits-ml-assignment-2`

## Live Streamlit App Link

Add your Streamlit Community Cloud link here after deployment:

`https://vgcqgkc7req59t9au8xump.streamlit.app/`

## d. Models used

The following classification models are implemented:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Gaussian Naive Bayes Classifier
- Random Forest Classifier

### Comparison table

Run the following command after installing dependencies to regenerate saved model artifacts and metrics:

```bash
python model/train_models.py
```

Baseline metrics from the current stratified test split:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9469 | 0.9806 | 0.9286 | 0.9286 | 0.9286 | 0.8863 |
| Logistic Regression | 0.9469 | 0.9909 | 0.9737 | 0.8810 | 0.9250 | 0.8867 |
| kNN | 0.9204 | 0.9611 | 0.9714 | 0.8095 | 0.8831 | 0.8313 |
| Decision Tree | 0.9027 | 0.8573 | 0.8974 | 0.8333 | 0.8642 | 0.7898 |
| Naive Bayes | 0.8938 | 0.9795 | 0.8947 | 0.8095 | 0.8500 | 0.7704 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Logistic Regression achieved the best AUC and the highest precision, showing that the scaled numeric features separate the benign and malignant classes well. Its recall is slightly lower than Random Forest. |
| Decision Tree | Decision Tree produced usable results but had the lowest AUC, suggesting that a single tree is less stable for this dataset than scaled linear models or ensembles. |
| kNN | kNN performed well after scaling and had high precision, but its recall was lower, meaning it missed more malignant cases than the best models. |
| Naive Bayes | Naive Bayes delivered strong AUC but lower threshold-based Accuracy, F1, and MCC, likely because the feature independence assumption is too simple for correlated cell measurements. |
| Random Forest (Ensemble) | Random Forest achieved the best F1 score and recall while matching Logistic Regression on accuracy, making it a balanced performer for this test split. |
| Overall Winner for this dataset? | Random Forest is the overall winner because it has the highest F1 score and recall while maintaining strong AUC and MCC. |

## Project structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- dataset.csv
|-- training_data.csv
|-- test_data.csv
|-- prepare_data.py
|-- raw_data/
|   |-- wdbc.data
|-- model/
|   |-- __init__.py
|   |-- model_utils.py
|   |-- train_models.py
|   |-- *.joblib
|   |-- metrics.csv
```

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Upload this folder to a GitHub repository.
2. Go to `https://streamlit.io/cloud`.
3. Create a new app from the GitHub repository.
4. Select the main branch.
5. Set the app entry point to `app.py`.
6. Deploy and copy the live app link into this README and the final PDF submission.

## Submission notes

The final PDF should include:

- GitHub repository link
- Live Streamlit app link
- BITS Virtual Lab execution screenshot
- README content
