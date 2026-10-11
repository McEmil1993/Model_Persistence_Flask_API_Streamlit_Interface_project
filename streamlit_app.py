import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass/Fail Predictor")
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass/Fail Predictor")

# Inputs
name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

# Prediction button
if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        X = pd.DataFrame({"study_hours": [hours]})
        result = int(model.predict(X)[0])
        pass_prob = float(model.predict_proba(X)[0][1])

        # Result message
        if result == 1:
            st.success(f"{name} is predicted to PASS.")
        else:
            st.error(f"{name} is predicted to FAIL.")

        # Short explanation
        st.write(
            f"**How this was generated:** The saved model was trained earlier on "
            f"study hours and exam results. It was loaded from a file (not retrained) "
            f"and given {hours} study hours as input. It estimated a {pass_prob:.0%} "
            f"chance of passing, so the prediction is "
            f"{'Pass' if result == 1 else 'Fail'}."
        )

        # Bonus: study hours vs. predicted class chart
        hours_range = [h / 2 for h in range(0, 33)]
        preds = model.predict(pd.DataFrame({"study_hours": hours_range}))
        chart = pd.DataFrame({"Predicted class (1 = Pass)": preds}, index=hours_range)
        chart.index.name = "Study hours"
        st.subheader("Study hours vs. predicted class")
        st.line_chart(chart)
        st.caption(f"{name}'s input: {hours} study hours")