"""Exercise 3 — Streamlit Interface."""
import joblib
import pandas as pd
import streamlit as st

MODEL_FILENAME = "student_pass_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILENAME)


st.title("Student Pass Predictor")
st.write("Enter a student's details to predict whether they will **Pass** or **Fail**.")

# 1. Student name input
student_name = st.text_input("Student name", placeholder="e.g. Angel")

# 2. Study-hours input
study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

# 3. Prediction button
if st.button("Predict"):
    if not student_name.strip():
        st.warning("Please enter a student name first.")
    else:
        try:
            model = load_model()
        except FileNotFoundError:
            st.error(f"Model file '{MODEL_FILENAME}' not found. Run train_model.py first.")
        else:
            pred = int(model.predict([[study_hours]])[0])
            proba = float(model.predict_proba([[study_hours]])[0].max())
            label = "Pass" if pred == 1 else "Fail"

            # 4. Result message
            if pred == 1:
                st.success(f"{student_name} is predicted to **PASS** (confidence: {proba:.0%}).")
            else:
                st.error(f"{student_name} is predicted to **FAIL** (confidence: {proba:.0%}).")

            # 5. Short explanation of how the prediction was generated
            st.info(
                f"How this was generated: the app loaded the pre-trained logistic-regression "
                f"model from `{MODEL_FILENAME}` (no retraining) and called "
                f"`model.predict([[{study_hours}]])` using study hours as the only feature. "
                f"The model compares the input against the decision boundary it learned "
                f"during training (~5 hours) and returns 1 = Pass or 0 = Fail."
            )

            # Bonus: chart comparing study hours with the predicted class
            st.subheader("Study hours vs predicted class")
            chart_data = pd.DataFrame({
                "Study hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float(study_hours)],
                "Predicted class": [int(model.predict([[h]])[0]) for h in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] + [pred],
                "Who": ["Training example"] * 10 + [student_name],
            })
            st.bar_chart(chart_data, x="Study hours", y="Predicted class")
            st.caption("0 = Fail, 1 = Pass. The last bar is the current student's prediction.")
else:
    st.caption("Fill in the fields above, then click **Predict**.")
