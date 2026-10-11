# Exercise 3 - Streamlit Interface (with Bonus chart)
import os

import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Student Pass Predictor", page_icon="🧠")


@st.cache_resource
def load_saved_model():
    return joblib.load(os.path.join(BASE, "student_pass_model.joblib"))


def as_frame(hours):
    # the model was trained with a column named "study_hours"
    return pd.DataFrame({"study_hours": hours})


model = load_saved_model()

st.title("🧠 Student Pass Predictor")
tab_predict, tab_about = st.tabs(["Predict", "About the model"])

with tab_about:
    st.write(
        "This app uses a model that was trained once, saved with joblib, and loaded "
        "here. It looks at the number of study hours and predicts Pass or Fail."
    )

with tab_predict:
    left, right = st.columns(2)
    student_name = left.text_input("Student name")
    study_hours = right.number_input("Study hours per day", 0.0, 24.0, 4.0, 0.5)

    if st.button("Predict"):
        if student_name.strip() == "":
            st.warning("Please enter the student's name.")
        else:
            chance = float(model.predict_proba(as_frame([study_hours]))[0][1])
            predicted = int(model.predict(as_frame([study_hours]))[0])

            # Result message
            if predicted == 1:
                st.success(f"{student_name} is likely to PASS.")
            else:
                st.error(f"{student_name} is likely to FAIL.")
            st.metric("Chance of passing", f"{chance:.0%}")

            # Short explanation
            st.info(
                "The model was trained on past examples of study hours and results. "
                f"It compared {study_hours} hour(s) with those examples and returned a "
                f"{chance:.0%} chance of passing, which gives the prediction above."
            )

            # Bonus chart: study hours vs predicted class
            hours_axis = [x / 2 for x in range(0, 25)]
            curve = pd.DataFrame(
                {
                    "Predicted class (1=Pass, 0=Fail)": model.predict(as_frame(hours_axis)),
                    "Chance of passing": model.predict_proba(as_frame(hours_axis))[:, 1],
                },
                index=pd.Index(hours_axis, name="Study hours"),
            )
            st.subheader("Study hours vs predicted class")
            st.line_chart(curve)
            st.caption(f"{student_name} entered {study_hours} hour(s) of study.")