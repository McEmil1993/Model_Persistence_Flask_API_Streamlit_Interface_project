import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")

# Load the saved model (no retraining needed)
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

st.title("🎓 Student Pass Predictor")
st.write("Enter a student's details to predict whether they will pass.")

# --- Inputs ---
student_name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5
)

# --- Prediction button ---
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        sample = pd.DataFrame({"study_hours": [study_hours]})
        prediction = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1])

        # --- Result message ---
        if prediction == 1:
            st.success(
                f"✅ {student_name} is predicted to PASS "
                f"(probability: {probability:.0%})."
            )
        else:
            st.error(
                f"❌ {student_name} is predicted to FAIL "
                f"(pass probability: {probability:.0%})."
            )

        # --- Short explanation ---
        st.subheader("How was this prediction generated?")
        st.write(
            f"The app loaded a Logistic Regression model that was trained earlier "
            f"on past study-hours data and saved with joblib. It took "
            f"**{study_hours} study hours** as input and calculated a pass "
            f"probability of **{probability:.0%}**. If the probability is 50% or "
            f"higher, the model predicts **Pass**; otherwise it predicts **Fail**."
        )

        # --- Bonus chart: study hours vs predicted class ---
        st.subheader("Chart: Study Hours vs Predicted Class")
        hours_range = np.linspace(0, 10, 100)
        grid = pd.DataFrame({"study_hours": hours_range})
        predicted_classes = model.predict(grid)
        probabilities = model.predict_proba(grid)[:, 1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.step(hours_range, predicted_classes, where="post",
                label="Predicted class (0 = Fail, 1 = Pass)", color="steelblue")
        ax.plot(hours_range, probabilities, "--",
                label="Pass probability", color="gray")
        ax.scatter([study_hours], [prediction], s=120,
                   color="green" if prediction == 1 else "red",
                   zorder=5, label=f"{student_name}")
        ax.set_xlabel("Study hours")
        ax.set_ylabel("Predicted class / probability")
        ax.set_yticks([0, 0.5, 1])
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)