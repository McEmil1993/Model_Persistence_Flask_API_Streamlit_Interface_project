from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Prediction")
st.write("Enter a student's details to view the model's prediction.")

if not MODEL_PATH.exists():
    st.error("Model file not found. First run: python train_model.py")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)

model = get_model()

student_name = st.text_input("Student name", placeholder="e.g., Juan Dela Cruz")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)

if st.button("Predict", type="primary"):
    prediction = int(model.predict([[study_hours]])[0])
    label = "Pass" if prediction == 1 else "Needs Improvement"
    display_name = student_name.strip() or "The student"

    if prediction == 1:
        st.success(f"{display_name}: predicted result — {label}")
    else:
        st.warning(f"{display_name}: predicted result — {label}")

    st.write(
        f"The model used **{study_hours:g} study hours** as input and returned "
        f"class **{prediction}** ({label}). The result is a model estimate, "
        "not a guarantee of actual academic performance."
    )
    st.info(
        "How it works: the app loads the previously trained model from "
        "`student_pass_model.joblib` and passes the entered study hours to it. "
        "The app does not train the model again for each prediction."
    )

    # Bonus chart: compare the entered study hours and predicted class.
    st.subheader("Study Hours and Predicted Class")
    st.caption("Class 0 = Needs Improvement; Class 1 = Pass.")
    fig, ax = plt.subplots()
    ax.bar(["Entered study hours", "Predicted class"], [study_hours, prediction])
    ax.set_ylabel("Value")
    ax.set_title("Input compared with model output")
    st.pyplot(fig)
