import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor")
st.title("Student Pass Predictor")

model = joblib.load("student_pass_model.joblib")

name = st.text_input("Student name")

hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=2.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])

        if pred == 1:
            st.success(f"{name} is likely to PASS.")
        else:
            st.error(f"{name} is likely to FAIL.")

        st.info(
            f"The model was trained on past examples of study hours and results. "
            f"For {hours} hour(s), it estimates a {prob:.0%} chance of passing. "
            f"If that is 50% or higher, it predicts Pass; otherwise Fail."
        )

        xs = np.arange(0, 12.5, 0.5)
        chart_df = pd.DataFrame({
            "Study hours": xs,
            "Predicted class (1=Pass, 0=Fail)": model.predict(xs.reshape(-1, 1)),
        }).set_index("Study hours")
        st.subheader("Study hours vs predicted class")
        st.line_chart(chart_df)
        st.caption(f"{name}'s input: {hours} hour(s) -> {'Pass' if pred == 1 else 'Fail'}")