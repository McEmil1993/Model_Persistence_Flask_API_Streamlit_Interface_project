from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓")
st.title("🎓 Student Success Predictor")
st.write("Estimate a student's outcome from study time and class attendance.")

MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(MODEL_FILE)
FEATURES = ["study_hours", "attendance_percentage"]

with st.form("prediction_form"):
    student_name = st.text_input("Student name", placeholder="e.g. Alex")
    study_hours = st.number_input("Study hours per week", min_value=0.0, max_value=80.0, value=5.0, step=0.5)
    attendance = st.slider("Attendance", min_value=0, max_value=100, value=80, format="%d%%")
    submitted = st.form_submit_button("Predict outcome")

if submitted:
    if not student_name.strip():
        st.warning("Enter a student name to see the result.")
    else:
        sample = pd.DataFrame(
            [[study_hours, float(attendance)]], columns=FEATURES
        )
        passed = bool(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1])
        if passed:
            st.success(f"{student_name.strip()} is predicted to pass (estimated pass probability: {probability:.0%}).")
        else:
            st.info(f"{student_name.strip()} may need additional support (estimated pass probability: {probability:.0%}).")
        st.caption(
            f"The saved classifier evaluated {study_hours:g} study hours per week and "
            f"{attendance}% attendance together. The model was trained on examples with both features."
        )

        attendance_values = [0, 25, 50, 75, 100]
        comparison = pd.DataFrame({
            "Attendance (%)": attendance_values,
            "Predicted class": model.predict(pd.DataFrame({
                "study_hours": [study_hours] * len(attendance_values),
                "attendance_percentage": attendance_values,
            })),
        })
        comparison["Predicted class"] = comparison["Predicted class"].map({0: "Needs improvement", 1: "Pass"})
        st.subheader("How attendance changes this prediction")
        st.write(f"Study hours held at {study_hours:g} per week")
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        chart, axis = plt.subplots()
        axis.plot(comparison["Attendance (%)"], comparison["Predicted class"].map({"Needs improvement": 0, "Pass": 1}), marker="o")
        axis.set_yticks([0, 1], ["Needs improvement", "Pass"])
        axis.set_xlabel("Attendance (%)")
        axis.set_title("Predicted class by attendance")
        axis.grid(axis="x", alpha=0.25)
        st.pyplot(chart)
