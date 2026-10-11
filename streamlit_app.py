"""Exercise 3: Streamlit interface."""
import joblib
import pandas as pd
import streamlit as st

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass/Fail Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        label = "Pass" if pred == 1 else "Fail"

        if pred == 1:
            st.success(f"{name} is predicted to PASS.")
        else:
            st.error(f"{name} is predicted to FAIL.")

        st.write(
            f"**How it was generated:** a Logistic Regression model was trained on past "
            f"study hours and results. It calculated a {prob:.0%} chance of passing for "
            f"{hours} study hours. If that chance is 50% or higher, it predicts Pass."
        )

        # Bonus chart: pass probability across study hours, with this student marked
        xs = [h / 2 for h in range(0, 49)]
        probs = model.predict_proba(pd.DataFrame({"study_hours": xs}))[:, 1]
        chart = pd.DataFrame({"study_hours": xs, "pass_probability": probs}).set_index("study_hours")
        st.line_chart(chart)
        st.caption(f"{name}: {hours} hours -> predicted class: {label}")
