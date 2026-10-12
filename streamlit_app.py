from pathlib import Path
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("Student Pass Predictor")
st.write("Enter a student's name and study hours to estimate the result.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

if st.button("Predict result"):
    result = int(model.predict([[hours]])[0])
    label = "Pass" if result else "Fail"
    student = name.strip() or "Student"
    if result:
        st.success(f"{student}, the model predicts: {label}")
    else:
        st.warning(f"{student}, the model predicts: {label}")
    st.write("The app loads the saved model and uses study hours to predict. It does not retrain the model for each prediction.")

st.subheader("Bonus: predicted class by study hours")
hours_range = np.arange(0, 13, 1).reshape(-1, 1)
predictions = model.predict(hours_range)
fig, ax = plt.subplots()
ax.plot(hours_range.ravel(), predictions, marker="o")
ax.set_xlabel("Study hours")
ax.set_ylabel("Predicted class (0 = Fail, 1 = Pass)")
ax.set_yticks([0, 1], ["Fail (0)", "Pass (1)"])
ax.set_xticks(range(0, 13, 2))
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)
st.info("This example model uses study hours only. Attendance is covered in the Flask API but does not affect this model.")
