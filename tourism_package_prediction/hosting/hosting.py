from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_package_prediction/deployment",     # the local folder containing your files
    repo_id="HF-Sum/tourism-package-prediction-space",          # IMPORTANT: Replace with your actual Hugging Face Space ID
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
