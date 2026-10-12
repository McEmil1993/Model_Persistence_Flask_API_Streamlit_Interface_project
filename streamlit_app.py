import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

st.title("🎓 Student Pass Predictor")

# Inputs
student_name = st.text_input("Student name")
study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

# Prediction button
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[study_hours]])[0])
        probability = float(model.predict_proba([[study_hours]])[0][1])

        # Result message
        if prediction == 1:
            st.success(f"{student_name} is predicted to PASS.")
        else:
            st.error(f"{student_name} is predicted to FAIL.")

        # Short explanation
        st.info(
            f"The model was trained earlier on example study hours and pass/fail results, "
            f"then loaded from a saved file. It used {study_hours} study hours to calculate "
            f"a pass probability of {probability:.0%}. Above 50% it predicts PASS, otherwise FAIL."
        )

        # Bonus chart: study hours vs predicted class
        hours_range = np.linspace(0, 12, 100).reshape(-1, 1)
        probabilities = model.predict_proba(hours_range)[:, 1]

        fig, ax = plt.subplots()
        ax.plot(hours_range, probabilities, label="Pass probability")
        ax.axhline(0.5, linestyle="--", color="gray", label="Decision threshold (0.5)")
        ax.scatter([study_hours], [probability], color="red", zorder=5,
                   label=f"{student_name}: {'PASS' if prediction == 1 else 'FAIL'}")
        ax.set_xlabel("Study hours")
        ax.set_ylabel("Probability of passing")
        ax.set_title("Study hours vs predicted class")
        ax.legend()
        st.pyplot(fig)