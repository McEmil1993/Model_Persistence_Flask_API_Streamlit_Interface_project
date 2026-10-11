# Exercise 3 
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "student_pass_model.joblib"
model = joblib.load(MODEL_FILE)

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")

col_name, col_hours = st.columns(2)
with col_name:
    student_name = st.text_input("Student name")
with col_hours:
    study_hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

clicked = st.button("Predict")

if clicked:
    if not student_name.strip():
        st.warning("Please type the student's name first.")
    else:
        probability = float(model.predict_proba([[study_hours]])[0][1])
        predicted_class = int(probability >= 0.5)   # 1 = Pass, 0 = Fail

        if predicted_class == 1:
            st.success(f"Result: {student_name} is likely to PASS.")
        else:
            st.error(f"Result: {student_name} is likely to FAIL.")
        st.metric("Chance of passing", f"{probability:.0%}")

        st.info(
            "How this was generated: the saved model was trained on past examples of "
            "study hours and pass/fail results. It looked at the hours you entered "
            f"({study_hours}), calculated a {probability:.0%} chance of passing, and "
            "predicts Pass when that chance is 50% or higher."
        )

        sample_hours = [h / 2 for h in range(0, 25)] 
        sample_classes = model.predict(pd.DataFrame(sample_hours).values)

        curve = pd.DataFrame({
            "Study hours": sample_hours,
            "Predicted class": ["Pass" if c == 1 else "Fail" for c in sample_classes],
        })
        you = pd.DataFrame({
            "Study hours": [study_hours],
            "Predicted class": ["Pass" if predicted_class == 1 else "Fail"],
            "Student": [student_name],
        })

        order = ["Fail", "Pass"]
        base_line = alt.Chart(curve).mark_line(interpolate="step-after", color="#4c78a8").encode(
            x=alt.X("Study hours:Q", title="Study hours"),
            y=alt.Y("Predicted class:N", sort=order, title="Predicted class"),
        )
        base_points = alt.Chart(curve).mark_circle(size=50, color="#4c78a8").encode(
            x="Study hours:Q",
            y=alt.Y("Predicted class:N", sort=order),
        )
        you_point = alt.Chart(you).mark_point(size=250, filled=True, color="red").encode(
            x="Study hours:Q",
            y=alt.Y("Predicted class:N", sort=order),
            tooltip=["Student", "Study hours", "Predicted class"],
        )

        st.subheader("Study hours vs predicted class")
        st.altair_chart(base_line + base_points + you_point, width="stretch")
        st.caption(f"Blue = model's prediction for each study-hours value. Red dot = {student_name} ({study_hours} hour(s)).")