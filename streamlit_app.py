"""Exercise 3 - Streamlit interface (with bonus chart)."""
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Study Hours Predictor", page_icon="📚")
model = joblib.load("model.pkl")  # study-hours-only model

st.title("📚 Student Pass/Fail Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=4.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        if pred == 1:
            st.success(f"{name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to FAIL (pass probability {prob:.0%}).")

        st.subheader("How was this prediction generated?")
        st.write(
            "A logistic regression model was trained earlier on past examples of "
            "study hours and pass/fail results, then saved with joblib. The app loads "
            f"that saved model and computes the probability of passing for {hours} hours. "
            "If the probability is 50% or higher, the predicted class is Pass; otherwise Fail."
        )

        # Bonus: study hours vs predicted class
        grid = np.linspace(0, 10, 100)
        curve = pd.DataFrame({
            "study_hours": grid,
            "pass_probability": model.predict_proba(pd.DataFrame({"study_hours": grid}))[:, 1],
        })
        point = pd.DataFrame({"study_hours": [hours], "pass_probability": [prob], "label": [name]})

        line = alt.Chart(curve).mark_line().encode(
            x=alt.X("study_hours", title="Study hours"),
            y=alt.Y("pass_probability", title="Probability of passing", scale=alt.Scale(domain=[0, 1])),
        )
        threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(y="y")
        dot = alt.Chart(point).mark_point(size=200, filled=True, color="red").encode(
            x="study_hours", y="pass_probability", tooltip=["label", "study_hours", "pass_probability"]
        )
        st.altair_chart(line + threshold + dot, use_container_width=True)
        st.caption("Red dot = this student. Dashed line = 50% cut-off between Fail and Pass.")
