import streamlit as slt
import pandas as pd
import joblib
import base64
import numpy as np

# Function to convert local image to Base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Convert image
image_base64 = get_base64_of_image(r"C:\Users\ZIYAD\Downloads\placement\background.avif")


# Set background using Base64
background_image = f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{image_base64}");
    background-size: cover;
    background-position: center;
}}
</style>
"""
slt.markdown(background_image, unsafe_allow_html=True)
custom_css = """
<style>
/* Style the checkbox labels */
div[data-testid="stCheckbox"] label {
    color: red !important;  /* Change text color */
    font-weight: bold;  /* Make text bold */
    font-size: 18px;  /* Adjust font size */
}

/* Change background of the checkbox */
div[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: blue !important;  /* Change checkbox color */
}
</style>
"""

# Apply the CSS
slt.markdown(custom_css, unsafe_allow_html=True)

slt.title("PLACEMENT PREDICTION")
model=joblib.load("placement_model.pkl")
cgpa=slt.number_input("CGPA",min_value=0,max_value=10)
internship=slt.number_input("Internship",min_value=0,max_value=10)
Projects=slt.number_input("Projects",min_value=0,max_value=10)
WorkshopsCertifications=slt.number_input("Workshops/Certifications",min_value=0,max_value=10)
AptitudeTestScore=slt.number_input("AptitudeTestScore",min_value=0,max_value=100)
SoftSkillsRating=slt.number_input("SoftSkillsRating",min_value=0,max_value=5)
ExtracurricularActivities=slt.radio("ExtracurricularActivities",["YES","No"])
PlacementTraining=slt.radio("PlacementTraining",["YES","No"])
sslc=slt.number_input("SSLC %",min_value=0,max_value=100)
puc=slt.number_input("PUC %",min_value=0,max_value=100)

extracurricular = 1 if ExtracurricularActivities == "Yes" else 0
placement_training = 1 if PlacementTraining== "Yes" else 0

if slt.button("Predict the Placement"):
    user_data=np.array([cgpa,internship,Projects,WorkshopsCertifications,AptitudeTestScore,SoftSkillsRating,extracurricular,placement_training,sslc,puc]).reshape(1,-1)
    user_data_prediction=model.predict(user_data)
    label = {
    0: "❌ Not Placed - Improve Your Profile",
    1: "✅ Placed - Congratulation"}
    slt.write(f"Placement Prediction: {label[user_data_prediction[0]]}")

