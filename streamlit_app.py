# streamlit_app.py - Exercise 3: Streamlit Interface
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_pass_model.joblib")


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


model = get_model()

st.set_page_config(page_title="Student Pass Predictor")
st.title("Student Pass Predictor")
st.write("Enter the student's details, then click Predict.")

with st.form("predict_form"):
    student = st.text_input("Student name")
    study_hours = st.slider("Study hours per day", 0.0, 12.0, 2.0, 0.5)
    submitted = st.form_submit_button("Predict")

if submitted:
    if student.strip() == "":
        st.warning("Please enter the student's name.")
    else:
        chance = model.predict_proba([[study_hours]])[0][1]
        will_pass = chance >= 0.5

        if will_pass:
            st.success(f"{student} is likely to PASS.")
        else:
            st.error(f"{student} is likely to FAIL.")

        st.info(
            "How it was generated: the model learned from past examples of study hours "
            f"and results. For {study_hours} hour(s) it calculated a {chance:.0%} chance "
            "of passing. 50% or higher means Pass, below 50% means Fail."
        )

        # Bonus chart: study hours vs predicted class
        hours_range = [x / 2 for x in range(0, 25)]
        classes = model.predict(pd.DataFrame(hours_range).values)
        chart_data = pd.DataFrame(
            {"Predicted class (1=Pass, 0=Fail)": classes}, index=hours_range
        )
        chart_data.index.name = "Study hours"
        st.subheader("Study hours vs predicted class")
        st.line_chart(chart_data)