import joblib
import pandas as pd
import streamlit as st

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
attendance = st.slider("Attendance percentage", 0, 100, 80)

if st.button("Predict"):
    X = pd.DataFrame([[hours, attendance]], columns=["study_hours", "attendance_percentage"])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    who = name.strip() or "This student"

    if pred == 1:
        st.success(f"{who} is predicted to PASS ({prob:.0%} pass probability).")
    else:
        st.error(f"{who} is predicted to FAIL ({prob:.0%} pass probability).")

    st.write(
        "**How this was generated:** a logistic regression model, trained earlier on past "
        "study hours and attendance, was loaded from `student_pass_model.joblib`. It combines "
        "the two inputs into a pass probability; 50% or higher is classified as Pass."
    )

    # Bonus: study hours vs predicted class (0-10 hours at the chosen attendance)
    sweep = pd.DataFrame({"study_hours": [h / 2 for h in range(0, 21)]})
    sweep["attendance_percentage"] = attendance
    sweep["predicted_class"] = model.predict(sweep[["study_hours", "attendance_percentage"]])
    st.subheader("Study hours vs predicted class (1 = Pass, 0 = Fail)")
    st.line_chart(sweep.set_index("study_hours")["predicted_class"])
