import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")

MODEL_FILE = "student_pass_model.joblib"

if not os.path.exists(MODEL_FILE):
    st.error(f"'{MODEL_FILE}' not found. Run `python train_model.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()

# ---- Inputs ----
student_name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)

# ---- Prediction ----
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        features = np.array([[study_hours]])
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        # Result message
        if prediction == 1:
            st.success(f"{student_name}, the model predicts you will PASS. ✅")
        else:
            st.error(f"{student_name}, the model predicts you will FAIL. ❌")

        # Explanation
        st.subheader("How was this prediction generated?")
        st.write(
            f"The model is a logistic regression trained on past examples of "
            f"study hours and results. It took your input of **{study_hours} hours** "
            f"and calculated a pass probability of **{probability:.0%}**. "
            f"Because that is {'at least' if prediction == 1 else 'below'} 50%, "
            f"it predicted **{'Pass' if prediction == 1 else 'Fail'}**. "
            f"The model was loaded from a saved file, so it was not retrained."
        )

        # Bonus chart: study hours vs. pass probability
        st.subheader("Study hours vs. predicted outcome")
        hours_range = np.linspace(0, 12, 100)
        probs = model.predict_proba(hours_range.reshape(-1, 1))[:, 1]
        curve = pd.DataFrame({"study_hours": hours_range, "pass_probability": probs})
        student_point = pd.DataFrame(
            {"study_hours": [study_hours], "pass_probability": [probability]}
        )

        line = (
            alt.Chart(curve)
            .mark_line()
            .encode(
                x=alt.X("study_hours", title="Study hours"),
                y=alt.Y(
                    "pass_probability",
                    title="Pass probability",
                    scale=alt.Scale(domain=[0, 1]),
                ),
            )
        )
        point = (
            alt.Chart(student_point)
            .mark_point(size=200, filled=True, color="red")
            .encode(x="study_hours", y="pass_probability")
        )
        st.altair_chart(line + point)
        st.caption("The red dot is this student. Above 0.5 means predicted Pass.")