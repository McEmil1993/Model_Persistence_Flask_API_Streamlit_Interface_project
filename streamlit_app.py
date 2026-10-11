import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("student_pass_model.joblib")

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("Student Pass Predictor")
st.write("Enter student details to get a pass/fail prediction.")

# Inputs
name = st.text_input("Student Name", placeholder="e.g. Alex")
study_hours = st.number_input(
    "Study Hours (per week)",
    min_value=0.0,
    max_value=40.0,
    value=5.0,
    step=0.5
)
attendance = st.slider(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=75
)

# Predict button
if st.button("Predict", type="primary"):
    if not name.strip():
        st.warning("Please enter a student name.")
    else:
        pred = model.predict([[study_hours, attendance]])[0]
        proba = model.predict_proba([[study_hours, attendance]])[0][1]
        label = "Pass" if pred == 1 else "Fail"

        # Result message
        if label == "Pass":
            st.success(
                f"**{name}** is predicted to **Pass** "
                f"with **{proba:.1%}** probability."
            )
        else:
            st.error(
                f"**{name}** is predicted to **Fail** "
                f"(only **{proba:.1%}** chance of passing)."
            )

        # Short explanation
        st.markdown("### How the prediction was generated")
        st.write(
            f"The model is a **logistic regression** trained on both "
            f"**study hours** and **attendance percentage**. "
            f"It learned a decision boundary from historical data. "
            f"For **{study_hours} hours** of study and **{attendance}%** attendance, "
            f"the model outputs a **{proba:.1%}** probability of passing. "
            f"The final class is chosen by thresholding at 50%."
        )

        # Bonus chart
        st.markdown("### Study Hours vs Predicted Class")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["Study Hours"], [study_hours], color="#4C72B0", width=0.5)
        ax.axhline(y=5, color="gray", linestyle="--", label="Typical decision boundary (~5h)")
        ax.set_ylabel("Hours")
        ax.set_ylim(0, max(12, study_hours + 2))
        ax.set_title(f"Predicted: {label}  |  Attendance: {attendance}%")
        ax.legend()
        st.pyplot(fig)