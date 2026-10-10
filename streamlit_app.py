import streamlit as st
import joblib
import pandas as pd

FEATURES = ["study_hours", "attendance_percentage"]
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass/Fail Predictor")

name = st.text_input("Student name")
study_hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
attendance = st.slider("Attendance (%)", 0, 100, 80)

if st.button("Predict"):
    X = pd.DataFrame([[study_hours, attendance]], columns=FEATURES)
    pred = int(model.predict(X)[0])
    prob_pass = float(model.predict_proba(X)[0][1])
    who = name if name else "The student"

    if pred == 1:
        st.success(f"{who} is predicted to PASS.")
    else:
        st.error(f"{who} is predicted to FAIL.")

    st.subheader("How was this prediction made?")
    st.write(
        f"The model was trained on past student records. It took {study_hours} study hours "
        f"and {attendance}% attendance, calculated a probability of passing of "
        f"**{prob_pass:.0%}**, and predicted 'Pass' if that probability is 50% or higher."
    )

    hours_range = [h / 2 for h in range(0, 25)]
    curve = pd.DataFrame({
        "study_hours": hours_range,
        "probability_of_passing": model.predict_proba(
            pd.DataFrame({"study_hours": hours_range,
                          "attendance_percentage": [attendance] * len(hours_range)})
        )[:, 1],
    })

    st.subheader(f"Pass probability vs. study hours (at {attendance}% attendance)")
    st.line_chart(curve, x="study_hours", y="probability_of_passing")

