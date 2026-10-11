import joblib
import pandas as pd
import streamlit as st

st.title("Student Pass Predictor")

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])

        if pred == 1:
            st.success(f"{name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to FAIL (probability of passing {prob:.0%}).")

        st.write(
            "**How this was generated:** a logistic regression model, trained earlier "
            "on past study hours and pass/fail results, was loaded from "
            "`student_pass_model.joblib`. It converts the study hours into a pass "
            "probability and predicts Pass if that is 50% or higher."
        )

        # Bonus: curve of P(pass) vs hours, with the student's point marked
        xs = [h / 2 for h in range(0, 49)]
        curve = pd.DataFrame({"Study hours": xs,
                              "P(pass)": model.predict_proba([[x] for x in xs])[:, 1]}).set_index("Study hours")
        st.line_chart(curve)
        st.caption(f"{name} studies {hours} h -> predicted class: {'Pass' if pred else 'Fail'} (1 = Pass, 0 = Fail)")