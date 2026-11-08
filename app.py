import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "diagnosis_malignant"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")
st.title("Breast Cancer Classification — Model Evaluation")
st.write(
    "Upload a held-out test CSV containing the 30 diagnostic features and "
    f"`{TARGET}` (1 = malignant, 0 = benign). Models are pre-trained; "
    "the uploaded data is used only for evaluation."
)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload the included `test_data.csv` to view evaluation results.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"Missing target column: {TARGET}")
    st.stop()

target = df[TARGET].astype(int)
features = df.drop(columns=[TARGET])
model_name = st.selectbox("Select classifier", list(MODEL_FILES))
model_path = MODEL_DIR / MODEL_FILES[model_name]
model = joblib.load(model_path)

expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
if expected is None:
    # Pipelines and estimators are trained with pandas feature names.
    expected = list(features.columns)
missing = sorted(set(expected) - set(features.columns))
extra = sorted(set(features.columns) - set(expected))
if missing or extra:
    st.error(f"Feature mismatch. Missing: {missing}; unexpected: {extra}")
    st.stop()

features = features[expected]
pred = model.predict(features)
prob = model.predict_proba(features)[:, 1]

metrics = {
    "Accuracy": accuracy_score(target, pred),
    "AUC": roc_auc_score(target, prob),
    "Precision": precision_score(target, pred, zero_division=0),
    "Recall": recall_score(target, pred, zero_division=0),
    "F1": f1_score(target, pred, zero_division=0),
    "MCC": matthews_corrcoef(target, pred),
}
st.subheader(model_name)
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(target, pred, labels=[0, 1])
    st.dataframe(pd.DataFrame(cm, index=["Actual benign", "Actual malignant"],
                              columns=["Predicted benign", "Predicted malignant"]))
with right:
    st.subheader("Classification report")
    st.code(classification_report(target, pred, target_names=["Benign", "Malignant"],
                                  zero_division=0))

st.subheader("Prediction sample")
result = features.copy()
result["actual"] = target
result["predicted"] = pred
result["malignant_probability"] = prob
st.dataframe(result.head(20), use_container_width=True)

if st.checkbox("Compare all five models"):
    rows = []
    for name, filename in MODEL_FILES.items():
        clf = joblib.load(MODEL_DIR / filename)
        p = clf.predict(features)
        s = clf.predict_proba(features)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(target, p),
            "AUC": roc_auc_score(target, s),
            "Precision": precision_score(target, p, zero_division=0),
            "Recall": recall_score(target, p, zero_division=0),
            "F1": f1_score(target, p, zero_division=0),
            "MCC": matthews_corrcoef(target, p),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(4), use_container_width=True)
