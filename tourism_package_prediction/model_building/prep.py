# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import OneHotEncoder # Using OneHotEncoder for categorical features
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_REPO_ID = "HF-Sum/tourism-package-prediction" # IMPORTANT: Replace <YourHFUsername> with your actual Hugging Face username
DATASET_FILE = "tourism.csv"

# Download and load the dataset from Hugging Face Hub
# This assumes tourism.csv is directly in the dataset repo root
from huggingface_hub import hf_hub_download

dataset_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=DATASET_FILE, repo_type="dataset")
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")

# Data cleaning as per the notebook
df.drop(["Unnamed: 0","CustomerID"], axis=1, inplace=True)

# Identify categorical and numerical columns for preprocessing
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Target column
target_col = 'ProdTaken'

# Remove target from numerical features list if it's there
if target_col in numerical_features:
    numerical_features.remove(target_col)

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Convert categorical features to dummy variables (One-Hot Encoding)
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify for imbalanced target
)

# Ensure output directory exists
os.makedirs("tourism_package_prediction/model_building/processed_data", exist_ok=True)

Xtrain.to_csv("tourism_package_prediction/model_building/processed_data/Xtrain.csv",index=False)
Xtest.to_csv("tourism_package_prediction/model_building/processed_data/Xtest.csv",index=False)
ytrain.to_csv("tourism_package_prediction/model_building/processed_data/ytrain.csv",index=False)
ytest.to_csv("tourism_package_prediction/model_building/processed_data/ytest.csv",index=False)


files_to_upload = [
    "tourism_package_prediction/model_building/processed_data/Xtrain.csv",
    "tourism_package_prediction/model_building/processed_data/Xtest.csv",
    "tourism_package_prediction/model_building/processed_data/ytrain.csv",
    "tourism_package_prediction/model_building/processed_data/ytest.csv"
]

# Upload processed data to Hugging Face dataset space
for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="processed_data/" + file_path.split("/")[-1],  # Upload to a subfolder named 'processed_data'
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )
print("Processed data uploaded to Hugging Face Hub.")
