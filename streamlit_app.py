import joblib
import pandas as pd
import streamlit as st

st.title("Student Pass Predictor")
model = joblib.load("student_pass_model.joblib")

name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        prob = model.predict_proba([[hours]])[0][1]
        label = "PASS" if pred == 1 else "FAIL"

        # Result message
        message = f"{name} is predicted to {label} (pass probability {prob:.0%})."
        if pred == 1:
            st.success(message)
        else:
            st.error(message)

        # Short explanation
        st.write(
            "**How this was generated:** a logistic regression model, trained earlier on "
            "study hours vs. pass/fail and loaded from `student_pass_model.joblib`, "
            f"estimated a {prob:.0%} chance of passing for {hours} hours. "
            "Above 50% means PASS."
        )

        # Bonus chart: study hours vs predicted class
        xs = [h / 2 for h in range(0, 49)]
        df = pd.DataFrame({
            "Study hours": xs,
            "Predicted class (1=Pass)": model.predict([[x] for x in xs]),
        }).set_index("Study hours")
        st.line_chart(df)
        st.caption(f"{name}'s input: {hours} hours -> class {pred}")