"""Exercise 3 - Streamlit interface (uses the Exercise 1 model: study hours only)."""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor")
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass/Fail Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0,
                        value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        sample = pd.DataFrame({"study_hours": [hours]})
        pred = int(model.predict(sample)[0])
        prob = float(model.predict_proba(sample)[0][1])
        label = "Pass" if pred == 1 else "Fail"

        if pred == 1:
            st.success(f"{name} is predicted to **{label}** ({prob:.0%} chance of passing).")
        else:
            st.error(f"{name} is predicted to **{label}** ({prob:.0%} chance of passing).")

        st.subheader("How was this prediction generated?")
        st.write(
            f"The app loaded a saved Logistic Regression model "
            f"(`student_pass_model.joblib`) that was trained earlier on past study "
            f"hours and exam results. It took the {hours} study hours you entered, "
            f"calculated the probability of passing ({prob:.0%}), and predicted "
            f"**Pass** if that probability is 50% or higher, otherwise **Fail**. "
            f"No retraining happened."
        )

        # Bonus chart: study hours vs. predicted class
        st.subheader("Bonus: Study hours vs. predicted class")
        hours_range = [h / 2 for h in range(0, 21)]
        preds = model.predict(pd.DataFrame({"study_hours": hours_range}))
        chart_df = pd.DataFrame({
            "Study hours": hours_range,
            "Predicted class (1 = Pass, 0 = Fail)": preds,
        }).set_index("Study hours")
        st.line_chart(chart_df)
        st.caption(f"{name}'s input ({hours} hours) is predicted as {label}.")
