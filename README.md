# ML Assignment 2 — Breast Cancer Classification

## a. Problem statement
Build and evaluate five supervised classification algorithms to classify breast tumour samples as **malignant (1)** or **benign (0)**. The Streamlit application evaluates saved, pre-trained models on an uploaded held-out CSV; it does not retrain models on uploaded test data.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), UCI Machine Learning Repository.
- **Local source:** `sklearn.datasets.load_breast_cancer` (the scikit-learn copy of the UCI dataset).
- **Instances:** 569; **input features:** 30 numeric features; **target:** `diagnosis_malignant`.
- **Target mapping:** 1 = malignant, 0 = benign.
- **Split:** 75% training / 25% test; stratified; `random_state=42`.
- **Test file:** `test_data.csv` contains the held-out test rows, all 30 features, and the target column.
- **Preprocessing:** StandardScaler is included inside the Logistic Regression, KNN, and SVM pipelines. Decision Tree and Gaussian Naive Bayes use the numeric feature values directly. The scaler is fitted only on training data and is saved as part of each pipeline.

## c. GitHub Repository Link
Add your repository URL here after creating the GitHub repository: [**[PASTE YOUR GITHUB REPOSITORY URL]**](https://github.com/RamkiDurai/BITS-2026AJ05363/)

## d. Models used

The six metrics below are calculated on the same held-out test set. AUC uses the predicted probability of the malignant class.

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.965  | 0.9962 |      0.98   |   0.9245 | 0.9515 | 0.9251 |
| Decision Tree          |     0.9441 | 0.9541 |      0.9787 |   0.8679 | 0.92   | 0.881  |
| K-Nearest Neighbors    |     0.958  | 0.986  |      0.9796 |   0.9057 | 0.9412 | 0.9103 |
| Gaussian Naive Bayes   |     0.9441 | 0.9952 |      1      |   0.8491 | 0.9184 | 0.883  |
| Support Vector Machine |     0.972  | 0.9966 |      1      |   0.9245 | 0.9608 | 0.9408 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.965, AUC 0.996, F1 0.951, MCC 0.925. |
| Decision Tree | Accuracy 0.944, AUC 0.954, F1 0.920, MCC 0.881. |
| K-Nearest Neighbors | Accuracy 0.958, AUC 0.986, F1 0.941, MCC 0.910. |
| Gaussian Naive Bayes | Accuracy 0.944, AUC 0.995, F1 0.918, MCC 0.883. |
| Support Vector Machine | Accuracy 0.972, AUC 0.997, F1 0.961, MCC 0.941. |
| **Overall winner** | **Support Vector Machine** has the highest F1 score in this run (AUC 0.9966, F1 0.9608). Select the model based on the metric priorities of the application; for screening contexts, malignant-class recall is especially important. |

## Repository structure

```text
aiml_assignment2_project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── evaluation_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app. Choose a model to see Accuracy, AUC, Precision, Recall, F1, MCC, confusion matrix, and classification report. Use the comparison checkbox to evaluate all five saved models on the same uploaded test data.

## Rebuild models

```bash
python train_models.py
```

This recreates the saved models, `test_data.csv`, and `evaluation_metrics.csv` using the fixed stratified split.

## Deployment notes
1. Push the complete project folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py`.
3. After deployment, add the GitHub repository URL and live Streamlit app URL to the final submission PDF.
4. Capture the required screenshot while running the assignment in BITS Virtual Lab. This project package cannot provide that proof screenshot or create your personal GitHub/Streamlit links.

## Academic integrity and reproducibility
This is a starter implementation for learning and customization. Review the code, understand the dataset and evaluation, and make your own modifications and observations before submission. The assignment brief states that AI tools are for learning support, not direct copy-paste submissions.
