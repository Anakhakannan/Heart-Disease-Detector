import streamlit as st
import pickle
import numpy  as np
import os.path as path
import logging

# Set page configuration
st.set_page_config(
    page_title='Heart Disease Detection',
    page_icon='🫀',
    layout='centered',
    initial_sidebar_state='auto'
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Custom CSS for background color and button
st.markdown(
    r"""
    <style>
    .stApp {
        background-color:  #FFB6C1; /* Light Pink */
    }
    .stButton>button {
        background-color: #FF0000; /* Red button color */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model_path = 'models/rf_model.pkl'
if not path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    logging.error(f"Model file not found at {model_path}")
else:
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logging.error(f"Error loading model: {e}")

# Title of the app
st.title('🫀Heart Disease Detection')

# Input features
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=25)
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
    restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
    ca = st.selectbox('No of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3, 4])
with col2:
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200)
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
with col3:
    cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
    thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

# Prediction
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    if st.button('Detect'):
        try:
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = model.predict(features)
            if prediction[0] == 1:
                st.markdown('<p style="color:red;">You have detected heart disease.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:green;">Congratulations! You have a healthy heart.</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            logging.error(f"Error making prediction: {e}")
