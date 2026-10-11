# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction")

name = st.text_input("Student Name")
hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0)
attendance = st.slider("Attendance Percentage", min_value=0, max_value=100, value=75)

if st.button("Predict"):
    pred = model.predict([[hours, attendance]])[0]
    result = "Pass" if pred == 1 else "Fail"

    st.success(f"{name or 'Student'}, prediction: **{result}**")
    st.info(f"The model used {hours} study hours and {attendance}% attendance as "
            f"input and compared them against patterns learned during training to "
            f"classify Pass/Fail.")

# Bonus chart: predicted pass probability across study hours at the chosen attendance
chart_data = pd.DataFrame({
    "Hours": list(range(1, 9)),
    "Pass Probability": [model.predict_proba([[h, attendance]])[0][1] for h in range(1, 9)]
})
st.bar_chart(chart_data.set_index("Hours"))
st.markdown(f"**Your input ({hours} hours, {attendance}% attendance) can be compared "
            f"against the probabilities above.**")
