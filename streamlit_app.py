"""Streamlit interface for the student pass prediction model."""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        from train_model import train_and_save_model

        train_and_save_model()
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")
st.write("Estimate whether a student will pass based on weekly study hours.")

student_name = st.text_input("Student name", placeholder="Enter the student's name")
study_hours = st.number_input("Study hours per week", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

if st.button("Predict result", type="primary"):
    if not student_name.strip():
        st.warning("Please enter a student name.")
    else:
        model = get_model()
        prediction = int(model.predict([[study_hours]])[0])
        predicted_class = "Pass" if prediction == 1 else "Fail"
        st.success(f"{student_name}, the predicted result is: **{predicted_class}**.")
        st.info(
            "The saved logistic-regression model compares the entered study hours with patterns learned "
            "from the training examples. It then chooses the more likely class: Pass or Fail."
        )

        # Show the model's pattern over a useful range instead of a single,
        # oversized bar. The highlighted point is the student's prediction.
        maximum_hours = min(20, max(10, int(study_hours) + 2))
        hours_to_compare = list(range(maximum_hours + 1))
        pass_probabilities = model.predict_proba([[hour] for hour in hours_to_compare])[:, 1]
        chart_data = pd.DataFrame(
            {
                "Study hours per week": hours_to_compare,
                "Chance of passing": pass_probabilities,
            }
        )
        selected_data = pd.DataFrame(
            {
                "Study hours per week": [study_hours],
                "Chance of passing": [model.predict_proba([[study_hours]])[0][1]],
            }
        )

        st.subheader("How study hours affect the prediction")
        st.caption("The dot marks this student's entered study hours.")
        line = alt.Chart(chart_data).mark_line(color="#4da3ff", strokeWidth=3).encode(
            x=alt.X("Study hours per week:Q", title="Study hours per week"),
            y=alt.Y(
                "Chance of passing:Q",
                title="Chance of passing",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("Study hours per week:Q", format=".0f"),
                alt.Tooltip("Chance of passing:Q", format=".1%"),
            ],
        )
        selected_point = alt.Chart(selected_data).mark_point(color="#ffb000", filled=True, size=150).encode(
            x="Study hours per week:Q",
            y="Chance of passing:Q",
            tooltip=[
                alt.Tooltip("Study hours per week:Q", format=".1f"),
                alt.Tooltip("Chance of passing:Q", format=".1%"),
            ],
        )
        st.altair_chart(line + selected_point, use_container_width=True)
