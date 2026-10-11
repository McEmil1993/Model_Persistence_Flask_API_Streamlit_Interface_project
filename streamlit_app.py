import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", layout="centered")

# Load model
model = joblib.load("student_pass_model.joblib")

st.title("Student Performance Predictor")

# Input widgets
student_name = st.text_input("Student Name", placeholder="e.g. Alex Smith")
study_hours = st.number_input(
    "Study Hours", min_value=0.0, max_value=24.0, value=4.0, step=0.5
)

# Prediction button
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter a student name.")
    else:
        # Run prediction
        features = np.array([[study_hours]])
        prediction = model.predict(features)[0]
        result = "Pass" if prediction == 1 else "Fail"

        # Result display
        if result == "Pass":
            st.success(f"Result for {student_name}: **{result}** 🎉")
        else:
            st.error(f"Result for {student_name}: **{result}**")

        # Short explanation
        st.info(
            f"**How this prediction was generated:**\n"
            f"The application fed the input value ({study_hours} study hours) into the "
            f"pre-trained Logistic Regression model (`student_pass_model.joblib`). The model evaluated "
            f"the learned decision threshold and classified the outcome as **{result}**."
        )

        # Bonus: Comparison Chart
        st.subheader("Study Hours vs. Predicted Class")

        # Generate range of study hours for curve visualization
        sample_hours = np.linspace(0, 12, 100).reshape(-1, 1)
        sample_preds = model.predict(sample_hours)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(
            sample_hours,
            sample_preds,
            label="Decision Boundary",
            color="#4CAF50",
            linewidth=2,
        )
        ax.scatter(
            [study_hours],
            [prediction],
            color="red",
            s=100,
            zorder=5,
            label=f"{student_name} ({study_hours} hrs)",
        )

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fail (0)", "Pass (1)"])
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Predicted Class")
        ax.set_title("Decision Curve & Current Student Position")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        st.pyplot(fig)