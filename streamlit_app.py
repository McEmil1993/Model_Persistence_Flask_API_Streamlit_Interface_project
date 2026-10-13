import joblib
import pandas as pd
import streamlit as st

st.title("Student Pass Predictor")

model = joblib.load("student_pass_model.joblib")

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
            st.error(f"{name} is predicted to FAIL (pass probability {prob:.0%}).")

        st.info(
            "How this was generated: a logistic regression model, trained earlier on "
            "past study hours and pass/fail outcomes, turns the hours you entered into "
            "a pass probability. If the probability is above 50%, the predicted class is Pass."
        )

        # Bonus: chart of the model's pass probability across study hours
        hours_range = [h / 2 for h in range(0, 25)]  # 0 to 12 in 0.5 steps
        probs = model.predict_proba([[h] for h in hours_range])[:, 1]
        chart_df = pd.DataFrame({"Pass probability": probs}, index=hours_range)
        chart_df.index.name = "Study hours"

        st.subheader("Study hours vs. predicted pass probability")
        st.line_chart(chart_df)
        st.caption(f"{name}'s input: {hours} hours -> {prob:.0%} pass probability")