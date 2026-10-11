import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(BASE_DIR, "student_pass_model.joblib")
LOGO_FILENAME = os.path.join(BASE_DIR, "tmclogo2.png")
PHOTO_FILENAME = os.path.join(BASE_DIR, "me.jpg")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILENAME)


model = load_model()

st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon=LOGO_FILENAME,
    layout="wide",
)

with st.sidebar:
    if os.path.exists(LOGO_FILENAME):
        st.image(LOGO_FILENAME, width=170)
    if os.path.exists(PHOTO_FILENAME):
        st.image(PHOTO_FILENAME, width=170)
    st.markdown("### Joshua B. Negre")
    st.markdown("**BSIT-4 — Block 9**")
    st.markdown("Trinidad Municipal College")
    st.divider()

header_logo, header_text = st.columns([1, 5])
with header_logo:
    if os.path.exists(LOGO_FILENAME):
        st.image(LOGO_FILENAME, width=110)
with header_text:
    st.title("🎓 Student Pass Prediction")
    st.write("Enter a student's study hours to estimate whether they will pass.")

student_name = st.text_input("Student name", placeholder="e.g. Juan Dela Cruz")
study_hours = st.number_input(
    "Study hours per week",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
)

if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student name before predicting.")
    else:
        features = np.array([[study_hours]])
        prediction = int(model.predict(features)[0])
        pass_probability = float(model.predict_proba(features)[0][1])

        if prediction == 1:
            st.success(
                f"{student_name} is predicted to **PASS** "
                f"(estimated pass probability: {pass_probability:.1%})."
            )
        else:
            st.error(
                f"{student_name} is predicted to **FAIL** "
                f"(estimated pass probability: {pass_probability:.1%})."
            )

        st.subheader("How this prediction was generated")
        st.write(
            "The saved logistic regression model was loaded from "
            f"`{MODEL_FILENAME}`. It applies the study-hour weights learned during "
            "training to the value you entered, then uses a 0.5 decision threshold: "
            "a pass probability above 0.5 is classified as PASS, otherwise FAIL. "
            "No retraining happens here - the model only applies the weights it "
            "already learned."
        )

        st.subheader("Study hours vs. predicted class")
        hours_axis = np.arange(0, 11, 0.5)
        predicted_classes = model.predict(hours_axis.reshape(-1, 1)).astype(int)
        chart_data = pd.DataFrame(
            {"study_hours": hours_axis, "predicted_class": predicted_classes}
        )
        bars = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("study_hours:O", title="study hours"),
                y=alt.Y(
                    "predicted_class:Q",
                    title="predicted class",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=["study_hours", "predicted_class"],
            )
        )
        input_marker = (
            alt.Chart(pd.DataFrame({"study_hours": [study_hours], "predicted_class": [prediction]}))
            .mark_point(color="red", size=160, filled=True)
            .encode(x="study_hours:O", y="predicted_class:Q", tooltip=["study_hours", "predicted_class"])
        )
        st.altair_chart((bars + input_marker).properties(height=320), use_container_width=True)
        st.caption(
            f"The red dot marks your input of {study_hours} study hours per week, "
            f"which falls in the {'PASS' if prediction == 1 else 'FAIL'} region "
            "(0 = fail, 1 = pass)."
        )
