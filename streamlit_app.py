# Exercise 3 - Streamlit interface
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass/Fail Predictor")

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])
        label = "Pass" if pred == 1 else "Fail"

        if pred == 1:
            st.success(f"{name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to FAIL (pass probability {prob:.0%}).")

        st.info(
            f"How this was generated: the saved model (student_pass_model.joblib) was loaded, "
            f"not retrained. It received {hours} study hours as input, calculated the probability "
            f"of passing using patterns learned from past student data, and predicted "
            f"'{label}' because that probability is {'at least' if pred == 1 else 'below'} 50%."
        )

        # Bonus chart: study hours vs. predicted class
        hrs = [h / 2 for h in range(0, 41)]
        probs = [float(model.predict_proba([[h]])[0][1]) for h in hrs]
        chart_df = pd.DataFrame({"Study hours": hrs, "Pass probability": probs}).set_index("Study hours")
        st.subheader("Study hours vs. predicted pass probability")
        st.line_chart(chart_df)
        st.caption(f"{name}'s input: {hours} hours -> {label}")
