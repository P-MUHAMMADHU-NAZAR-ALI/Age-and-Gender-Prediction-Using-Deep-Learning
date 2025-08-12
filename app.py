import streamlit as st
from PIL import Image
from transformers import pipeline
import numpy as np
import librosa
import joblib
import subprocess
import tempfile
import os

# Dynamically get the base directory (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the age detection model
age_pipe = pipeline("image-classification", model="ares1123/photo_age_detection")

# Load the gender detection model for images
gender_pipe = pipeline("image-classification", model="rizvandwiki/gender-classification")

# Load the trained XGBoost model for gender recognition from voice
gender_classes = ['Male', 'Female']
model_path = os.path.join(BASE_DIR, "Gender-Recognition-from-Voice", "xgboost_model.pkl")
gender_model = joblib.load(model_path)

# Define CSS styles
STYLE = """
<style>
.upload-button {
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
}
.result-container {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
.result {
    font-size: 18px;
    font-weight: bold;
    color: #333333;
}
</style>
"""

def main():
    st.set_page_config(page_title="Age and Gender Detection App", layout="centered", initial_sidebar_state="auto")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown("<h1>Age and Gender Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<p>Upload a photo to determine the age and gender of the person.</p>", unsafe_allow_html=True)

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="upload_image")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Analyze button
        if st.button("Analyze Image"):
            predicted_age, predicted_gender = detect_age_and_gender(image)
            display_results(predicted_age, predicted_gender)

    # Upload voice
    uploaded_voice = st.file_uploader("Choose an audio file...", type=["wav", "mp3"], key="upload_voice")

    if uploaded_voice is not None:
        # Create a temporary file for the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_voice.read())
            temp_file_path = temp_file.name

        # Analyze button for voice
        if st.button("Analyze Voice"):
            predicted_voice_gender = detect_voice_gender(temp_file_path)
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown(f'<p class="result">Predicted Voice Gender: {predicted_voice_gender}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Clean up the temporary file after processing
        os.remove(temp_file_path)

# Function to detect age and gender using the provided models
def detect_age_and_gender(image):
    age_result = age_pipe(image)
    predicted_age = age_result[0]['label'].split('_')[-1]

    gender_result = gender_pipe(image)
    predicted_gender = gender_result[0]['label'].split('_')[-1]

    return predicted_age, predicted_gender

def detect_voice_gender(audio_file_path):
    """
    Perform gender recognition on the given audio file path.

    Args:
        audio_file_path (str): The path to the input audio file.

    Returns:
        str: Predicted gender or 'Error' if failed.
    """
    try:
        rscript_exe = "C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe"
        r_script_path = os.path.join(BASE_DIR, "Gender-Recognition-from-Voice", "voice_analysis.R")

        # Wrap full command in a string when using shell=True
        command = f'"{rscript_exe}" "{r_script_path}" "{audio_file_path}"'

        # Run the R script
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True  # Required to process the string correctly in Windows
        )

        if result.returncode != 0:
            st.error(f"R Script Error:\n{result.stderr}")
            return "Error"

        output = result.stdout.strip()
        if not output:
            st.error("No output from R script.")
            return "Error"

        features = [float(v) for v in output.split() if '.' in v or v.isdigit()]
        if len(features) != 20:
            st.error(f"Expected 20 features, got {len(features)}:\n{output}")
            return "Error"

        features = np.array(features).reshape(1, -1)
        pred_index = gender_model.predict(features)[0]
        pred_gender = gender_classes[int(pred_index)]

        return pred_gender

    except Exception as e:
        st.error(f"Python Exception: {str(e)}")
        return "Error"



def display_results(predicted_age, predicted_gender):
    with st.container():
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown(f'<p class="result">Predicted Age: {predicted_age}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="result">Predicted Gender: {predicted_gender}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
