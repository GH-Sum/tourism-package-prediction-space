import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Constants for Hugging Face Hub
MODEL_REPO_ID = "HF-Sum/tourism-package-prediction-model" # IMPORTANT: Replace with your actual Hugging Face model repo ID
MODEL_FILENAME = "best_tourism_package_model_v1.joblib"
DATASET_REPO_ID = "HF-Sum/tourism-package-prediction" # IMPORTANT: Replace with your actual Hugging Face dataset repo ID
XTRAIN_FILENAME = "processed_data/Xtrain.csv"

# Download and load the model
model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type="model")
model = joblib.load(model_path)

# Download Xtrain to get the exact column order and names for preprocessing
xtrain_columns_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=XTRAIN_FILENAME, repo_type="dataset")
xtrain_df_for_cols = pd.read_csv(xtrain_columns_path)
expected_columns = xtrain_df_for_cols.columns.tolist()

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase a **Wellness Tourism Package** based on their details.
Please enter the customer information below to get a prediction.
""")

# User input fields
st.header("Customer Details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Government"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2, step=1)
    numberoffollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=6, value=3, step=1)
    productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
with col2:
    preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
    maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    numberoftrips = st.number_input("Number of Trips Annually", min_value=1, max_value=25, value=3, step=1)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    owncar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    numberofchildrenvisiting = st.number_input("Number of Children (below 5)", min_value=0, max_value=3, value=0, step=1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Director"])
    monthlyincome = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=25000.0, step=100.0)
    durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=5.0, max_value=150.0, value=15.0, step=1.0)


# Assemble input into DataFrame
input_data_dict = {
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'DurationOfPitch': durationofpitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'NumberOfFollowups': numberoffollowups,
    'ProductPitched': productpitched,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyincome
}
input_df = pd.DataFrame([input_data_dict])

# Convert 'Passport' and 'OwnCar' to object/category type for one-hot encoding consistency
# with the training data (where they were converted to category and then one-hot encoded by get_dummies)
input_df['Passport'] = input_df['Passport'].astype(object)
input_df['OwnCar'] = input_df['OwnCar'].astype(object)

# Identify categorical features for one-hot encoding, matching the prep.py logic
categorical_features_for_ohe = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation', 'Passport', 'OwnCar'
]

# Apply one-hot encoding
input_df_processed = pd.get_dummies(input_df, columns=categorical_features_for_ohe, drop_first=True)

# Align columns with the training data columns
# Add missing columns from Xtrain and ensure order
for col in expected_columns:
    if col not in input_df_processed.columns:
        input_df_processed[col] = 0 # Add missing columns with 0

input_df_processed = input_df_processed[expected_columns] # Ensure correct column order


# Prediction
if st.button("Predict Purchase"):
    prediction_proba = model.predict_proba(input_df_processed)[:, 1][0]
    prediction = model.predict(input_df_processed)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"**Customer is likely to purchase the Wellness Tourism Package!** (Probability: {prediction_proba:.2f})")
    else:
        st.info(f"**Customer is unlikely to purchase the Wellness Tourism Package.** (Probability: {prediction_proba:.2f})")

    st.write(f"Raw Prediction: {prediction}")
