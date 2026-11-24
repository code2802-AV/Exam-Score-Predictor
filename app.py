import streamlit as st
import pandas as pd
import joblib

# Website title aur layout
st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓")
st.title("🎓 Student Exam Score Predictor")
st.write("Apne study details select karein aur Machine Learning model se expected final score check karein.")

# Trained model load karo
model = joblib.load('score_model.pkl')

# User inputs
study_hours = st.slider("Daily Study Hours", min_value=1.0, max_value=12.0, value=5.0, step=0.5)
attendance = st.slider("Attendance Percentage (%)", min_value=40, max_value=100, value=80)
prev_score = st.number_input("Previous Exam Score (out of 100)", min_value=0, max_value=100, value=65)

# Prediction button
if st.button("Predict Final Score"):
    input_df = pd.DataFrame(
        [[study_hours, attendance, prev_score]],
        columns=['Study_Hours', 'Attendance', 'Previous_Score']
    )
    prediction = model.predict(input_df)[0]
    final_score = max(0, min(100, round(prediction, 2)))

    st.success(f"Predicted Final Score: **{final_score} / 100**")
    st.progress(int(final_score))