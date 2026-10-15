"""Exercise 3: Streamlit interface."""
import joblib
import pandas as pd
import streamlit as st

st.title("Student Pass Predictor")

model = joblib.load("student_pass_model.joblib")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

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

        st.write(
            "The model was trained earlier on examples of study hours and pass/fail results. "
            "It learned how study hours relate to passing, and now uses that pattern on "
            f"{hours} hours to estimate the chance of passing. Above 50% means 'pass'."
        )

        # Bonus: study hours vs predicted class
        hours_range = [h / 2 for h in range(0, 25)]
        curve = pd.DataFrame({
            "study_hours": hours_range,
            "pass_probability": model.predict_proba(pd.DataFrame({"study_hours": hours_range}))[:, 1],
        }).set_index("study_hours")
        st.subheader("Study hours vs. chance of passing")
        st.line_chart(curve)
        st.caption(f"{name}: {hours} hours -> {'pass' if pred else 'fail'}")