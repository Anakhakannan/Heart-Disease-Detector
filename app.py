import streamlit as st
import pickle
import numpy as np
import os.path as path

# Load the trained model
file_name = "rf_model.pkl"
with open(path.join("model", file_name), 'rb') as file:
    rf_model = pickle.load(file)

# Title of the app
st.set_page_config(
    page_title='Heart Disease Detection',
    page_icon='❤️',
    layout='centered',
    initial_sidebar_state='auto'
)
st.title('❤️Heart Disease Detection')

# Input features
col1,col2 ,col3 = st.columns(3)
with col1:
     age = st.number_input('Age', min_value=1, max_value=120, value=25)
with col2:
     sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
with col3:
     cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
with col1:
     trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
with col2:
     chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200)
with col3:
     fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
with col1:
     restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
with col2:
     thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
with col3:
     exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
with col1:
     oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
with col2:
     slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
with col1:
     ca = st.selectbox('No of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3, 4])
with col3:
     thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

# Prediction
if st.button('Detect Heart Disease'):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = rf_model.predict(features)
    if prediction[0] == 1:
        st.write('The person has Detected heart disease.')
    else:
        st.write('The person does not have heart disease.')
