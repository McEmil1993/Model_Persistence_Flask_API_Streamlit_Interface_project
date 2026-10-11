import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Two-feature model (study hours + attendance)
model = joblib.load("student_pass_model_v2.joblib")

st.title("Student Pass Predictor")

name = st.text_input("Student name")
hours = st.number_input(
    "Study hours per day",
    min_value=0.0, max_value=24.0, value=3.0, step=0.5,
)
attendance = st.slider("Attendance percentage", min_value=0, max_value=100, value=75)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame(
            [[hours, attendance]],
            columns=["study_hours", "attendance_percentage"],
        )
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        if pred == 1:
            st.success(f"{name} is predicted to PASS ({prob:.0%} probability).")
        else:
            st.error(f"{name} is predicted to FAIL ({prob:.0%} probability of passing).")

        st.info(
            "The app loaded a model trained earlier on past student data. "
            f"It used {hours} study hours and {attendance}% attendance, "
            "applied the patterns it learned, and returned the most likely "
            "class and its probability."
        )

        # Bonus chart: probability vs study hours at this student's attendance
        xs = np.linspace(0, 10, 100)
        grid = pd.DataFrame({
            "study_hours": xs,
            "attendance_percentage": attendance,
        })
        probs = model.predict_proba(grid)[:, 1]

        fig, ax = plt.subplots()
        ax.plot(xs, probs, label=f"Probability of passing ({attendance}% attendance)")
        ax.axhline(0.5, linestyle="--", color="gray", label="Pass threshold (0.5)")
        ax.scatter([hours], [prob], color="red", zorder=3, label=name)
        ax.set_xlabel("Study hours")
        ax.set_ylabel("Probability of passing")
        ax.legend()
        st.pyplot(fig)