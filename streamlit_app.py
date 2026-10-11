import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

st.title("🎓 Student Pass Predictor")
st.write("Enter a student's details to predict whether they are likely to pass.")

# --- Inputs ---
student_name = st.text_input("Student name")
study_hours = st.number_input(
    "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
)

# --- Prediction button ---
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter the student's name first.")
    else:
        sample = pd.DataFrame({"study_hours": [study_hours]})
        prediction = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1])

        # --- Result message ---
        if prediction == 1:
            st.success(
                f"✅ {student_name} is predicted to PASS "
                f"(pass probability: {probability:.0%})."
            )
        else:
            st.error(
                f"❌ {student_name} is predicted to FAIL "
                f"(pass probability: {probability:.0%})."
            )

        # --- Explanation ---
        coef = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])
        st.subheader("How was this prediction generated?")
        st.write(
            f"The saved model is a logistic regression trained on past study hours "
            f"and pass/fail results. It did not train again just now; it loaded its "
            f"learned values and applied them to the {study_hours} hours you entered."
        )
        st.latex(r"P(\text{pass}) = \frac{1}{1 + e^{-(b_0 + b_1 \cdot \text{hours})}}")
        st.write(
            f"With the learned values b₀ = {intercept:.2f} and b₁ = {coef:.2f}, "
            f"the pass probability is {probability:.2f}. If it is above 0.5, "
            f"the model predicts PASS; otherwise it predicts FAIL."
        )

        # --- Bonus chart: study hours vs predicted class ---
        st.subheader("Study hours vs predicted class")
        hours_range = np.linspace(0, 12, 200)
        curve = model.predict_proba(pd.DataFrame({"study_hours": hours_range}))[:, 1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hours_range, curve, label="Pass probability curve")
        ax.axhline(0.5, color="gray", linestyle="--", label="PASS / FAIL threshold (0.5)")
        ax.scatter(
            [study_hours],
            [probability],
            color="green" if prediction == 1 else "red",
            s=120,
            zorder=5,
            label=f"{student_name}: {'PASS' if prediction == 1 else 'FAIL'}",
        )
        ax.set_xlabel("Study hours")
        ax.set_ylabel("Pass probability")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        st.pyplot(fig)