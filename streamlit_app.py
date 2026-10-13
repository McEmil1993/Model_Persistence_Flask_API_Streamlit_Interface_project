from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="📊")


@st.cache_resource
def fetch_model():
    return joblib.load(MODEL_FILE)


def run_prediction(model, hours):
    """Return (label, chance_of_passing) for the given study hours."""
    x = np.array([[hours]])
    chance = float(model.predict_proba(x)[0][1])
    label = "PASS" if model.predict(x)[0] == 1 else "FAIL"
    return label, chance


model = fetch_model()

st.title("📊 Student Pass Predictor")

# 1 + 2. Inputs
name_col, hours_col = st.columns([2, 1])
student_name = name_col.text_input("Student name")
study_hours = hours_col.number_input("Study hours", min_value=0.0, max_value=24.0, value=4.0, step=0.5)

# 3. Prediction button
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        label, chance = run_prediction(model, study_hours)

        # 4. Result message
        message = f"{student_name} is likely to {label}."
        (st.success if label == "PASS" else st.error)(message)

        # 5. Short explanation
        st.info(
            "This prediction comes from a saved model that learned from past examples of "
            f"study hours and results. It compared {study_hours} hour(s) with the most similar "
            f"examples and found a {chance:.0%} chance of passing. If the chance is 50% or "
            "higher the result is Pass; otherwise it is Fail."
        )

        # Bonus: chart comparing study hours with the predicted class
        hours_grid = np.arange(0, 12.5, 0.5)
        classes = model.predict(hours_grid.reshape(-1, 1))
        chart_df = pd.DataFrame(
            {"Predicted class (1=Pass, 0=Fail)": classes},
            index=pd.Index(hours_grid, name="Study hours"),
        )
        st.subheader("Study hours vs predicted class")
        st.area_chart(chart_df)
        st.caption(f"{student_name} studied {study_hours} hour(s) and was predicted to {label}.")