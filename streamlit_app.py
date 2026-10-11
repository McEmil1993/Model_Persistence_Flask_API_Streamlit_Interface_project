import joblib
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from train_model import load_model

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Pass/Fail Predictor")
st.write("A simple classroom demonstration using study hours.")

model = load_model()

name = st.text_input("Student name", placeholder="Enter student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

if st.button("Predict", type="primary"):
    prediction = int(model.predict([[hours]])[0])
    result = "Pass" if prediction == 1 else "Fail"

    if name.strip():
        st.subheader(f"Result for {name.strip()}")
    else:
        st.subheader("Prediction result")

    if result == "Pass":
        st.success(f"Predicted class: {result}")
    else:
        st.warning(f"Predicted class: {result}")
    st.write(
        f"The model received **{hours:g} study hours** and used patterns learned from a small sample dataset "
        f"to predict **{result}**. This is a classroom example, not a reliable real-world assessment."
    )

    fig, ax = plt.subplots()
    ax.bar(["Study hours", "Predicted class (Pass=1, Fail=0)"], [hours, prediction])
    ax.set_title("Input and model prediction")
    ax.set_ylabel("Value")
    st.pyplot(fig)
    plt.close(fig)

st.caption("Demo only: the model uses study hours, not attendance, and should not be used for real academic decisions.")
