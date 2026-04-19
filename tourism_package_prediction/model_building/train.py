# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline # Use Pipeline directly for better readability with make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score # Added roc_auc_score for classification
)

# for model serialization
import joblib
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO_ID = "HF-Sum/tourism-package-prediction" # Your actual Hugging Face username/dataset-repo
MODEL_REPO_ID = "HF-Sum/tourism-package-prediction-model" # IMPORTANT: Replace <YourHFUsername> with your actual Hugging Face username

# Download and load the processed data from Hugging Face Hub
from huggingface_hub import hf_hub_download

Xtrain_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="processed_data/Xtrain.csv", repo_type="dataset")
Xtest_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="processed_data/Xtest.csv", repo_type="dataset")
ytrain_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="processed_data/ytrain.csv", repo_type="dataset")
ytest_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="processed_data/ytest.csv", repo_type="dataset")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze() # Ensure y is a Series
ytest = pd.read_csv(ytest_path).squeeze()   # Ensure y is a Series

# Identify numerical features to be scaled from the Xtrain dataframe
# These are the original numerical features before one-hot encoding categorical ones
numerical_features_for_scaling = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

# Preprocessing pipeline (only scaling numerical features, as categorical are already one-hot encoded in Xtrain/Xtest)
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features_for_scaling),
    remainder='passthrough' # Keep the one-hot encoded features as they are
)

# Define XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")

# Define hyperparameter grid for XGBClassifier
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.9],
    'xgbclassifier__colsample_bytree': [0.7, 0.9],
}

# Create pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('xgbclassifier', xgb_model)])

# Grid search with cross-validation using an appropriate scoring metric for classification
grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)
y_prob_train = best_model.predict_proba(Xtrain)[:, 1] # Probabilities for AUC
y_prob_test = best_model.predict_proba(Xtest)[:, 1]   # Probabilities for AUC

# Evaluation
print("\nTraining Performance:")
print("Accuracy:", accuracy_score(ytrain, y_pred_train))
print("Precision:", precision_score(ytrain, y_pred_train))
print("Recall:", recall_score(ytrain, y_pred_train))
print("F1-Score:", f1_score(ytrain, y_pred_train))
print("ROC AUC Score:", roc_auc_score(ytrain, y_prob_train))
print("Confusion Matrix:\n", confusion_matrix(ytrain, y_pred_train))

print("\nTest Performance:")
print("Accuracy:", accuracy_score(ytest, y_pred_test))
print("Precision:", precision_score(ytest, y_pred_test))
print("Recall:", recall_score(ytest, y_pred_test))
print("F1-Score:", f1_score(ytest, y_pred_test))
print("ROC AUC Score:", roc_auc_score(ytest, y_prob_test))
print("Confusion Matrix:\n", confusion_matrix(ytest, y_pred_test))

# Create model directory
os.makedirs("tourism_package_prediction/model_building/models", exist_ok=True)

# Save best model locally
model_filename = "best_tourism_package_model_v1.joblib"
joblib.dump(best_model, os.path.join("tourism_package_prediction/model_building/models", model_filename))
print(f"Model saved locally as {model_filename}")

# Upload to Hugging Face Model Hub
# Step 1: Check if the model repo exists
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
    print(f"Model Space '{MODEL_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{MODEL_REPO_ID}' not found. Creating new model space...")
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)
    print(f"Model Space '{MODEL_REPO_ID}' created.")

api.upload_file(
    path_or_fileobj=os.path.join("tourism_package_prediction/model_building/models", model_filename),
    path_in_repo=model_filename,
    repo_id=MODEL_REPO_ID,
    repo_type="model",
)
print(f"Model '{model_filename}' uploaded to Hugging Face Model Hub: {MODEL_REPO_ID}")
