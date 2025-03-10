import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("BreatheEZ: Pneumonia Detection System")

try:
    if "xray_model" not in st.session_state:
        st.session_state.xray_model = load_model("pneumonia_detection_model.keras")
    st.success("X-ray detection model loaded successfully.")
except Exception as e:
    st.error(f"Error loading Keras model: {e}")
    st.stop()

st.header("X-ray Based Prediction")
uploaded_file = st.file_uploader("Upload an X-ray Image (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
st.session_state.xray_result = None
st.session_state.xray_probability = 0.0

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        prediction = st.session_state.xray_model.predict(image_array)[0]
        st.session_state.xray_probability = prediction[0]
        st.session_state.xray_result = "Pneumonia Detected" if st.session_state.xray_probability > 0.5 else "No Pneumonia"
        
        st.write(f"X-ray Prediction Result: {st.session_state.xray_result} (Confidence: {st.session_state.xray_probability:.2%})")
    except Exception as e:
        st.error(f"Error processing image: {e}")


st.header("Symptom-Based Prediction")

pneumonia_symptoms = [
    "chills", "fatigue", "high fever", "rusty sputum", "phlegm", "cough", 
    "breathlessness", "malaise", "chest pain", "fast heart rate", "sweating"
]

if "symptom_model" not in st.session_state:
    try:
        data = pd.read_csv("symptom dataset.csv")
        data["Pneumonia"] = data["Disease"].str.contains("pneumonia", case=False, na=False).astype(int)
        
        symptom_columns = [col for col in data.columns if col.startswith("Symptom_")]
        data["All_Symptoms"] = data[symptom_columns].apply(lambda x: ','.join(x.dropna().str.strip().str.lower()), axis=1)
        all_symptoms = sorted(set(','.join(data["All_Symptoms"]).split(',')))
        
        for symptom in all_symptoms:
            data[symptom] = data["All_Symptoms"].str.contains(symptom, na=False).astype(int)
        
        X = data[all_symptoms]
        y = data["Pneumonia"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        symptom_model = RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced")
        symptom_model.fit(X_train, y_train)
        
        st.session_state.symptom_model = symptom_model
        st.session_state.all_symptoms = all_symptoms
        
        st.success("Symptom prediction model trained successfully.")
    except Exception as e:
        st.error(f"Error during symptom model training: {e}")
        st.stop()

selected_symptoms = st.multiselect("Select symptoms:", st.session_state.all_symptoms)

if st.button("Predict Symptoms"):
    if selected_symptoms:
        pneumonia_symptom_count = sum(1 for symptom in selected_symptoms if symptom in pneumonia_symptoms)
        
        if pneumonia_symptom_count >= 2:
            st.session_state.symptom_result = "High likelihood of pneumonia"
            st.session_state.symptom_probability = 1.0
        else:
            input_data = pd.DataFrame([[int(symptom in selected_symptoms) for symptom in st.session_state.all_symptoms]],
                                      columns=st.session_state.all_symptoms)
            
            prediction = st.session_state.symptom_model.predict(input_data)[0]
            st.session_state.symptom_probability = st.session_state.symptom_model.predict_proba(input_data)[0][1]
            
            if prediction == 1:
                st.session_state.symptom_result = f"High likelihood of pneumonia (Confidence: {st.session_state.symptom_probability:.2%})"
            else:
                st.session_state.symptom_result = f"Low likelihood of pneumonia (Confidence: {st.session_state.symptom_probability:.2%})"
        
        st.write(f"### Symptom-Based Prediction: {st.session_state.symptom_result}")
    else:
        st.warning("Please select at least one symptom before clicking Predict.")

st.header("Final Combined Prediction")

if st.session_state.xray_result and "symptom_result" in st.session_state:
    st.write(f"X-ray Model Result: {st.session_state.xray_result} (Confidence: {st.session_state.xray_probability:.2%})")
    st.write(f"Symptom Model Result: {st.session_state.symptom_result}")
    
    combined_score = 0.6 * st.session_state.xray_probability + 0.4 * st.session_state.symptom_probability
    if combined_score > 0.5:
        st.success("Final Diagnosis: High likelihood of pneumonia")
    else:
        st.info("Final Diagnosis: Low to moderate likelihood of pneumonia")
else:
    st.warning("Please provide both X-ray and symptom inputs for a combined prediction.")