"""Exercise 3 - Streamlit interface."""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")

model = joblib.load("student_pass_model.joblib")

st.title("🎓 Student Pass Predictor")
st.write("Enter a student's study hours to predict whether they will pass.")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        features = pd.DataFrame({"study_hours": [hours]})
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        if prediction == 1:
            st.success(f"✅ {name} is predicted to PASS (probability {probability:.0%}).")
        else:
            st.error(f"❌ {name} is predicted to FAIL (pass probability {probability:.0%}).")

        st.subheader("How was this prediction generated?")
        st.write(
            "A logistic regression model was trained on past student data. "
            "It turns the study hours into a pass probability. "
            "If that probability is 50% or higher, the predicted class is **pass** (1); "
            "otherwise it is **fail** (0). The saved model was loaded from "
            "`student_pass_model.joblib`, so nothing was retrained."
        )

        # Bonus: chart comparing study hours with the predicted class
        st.subheader("Study hours vs predicted class")
        hours_range = [h / 2 for h in range(0, 25)]
        X_curve = pd.DataFrame({"study_hours": hours_range})
        curve = X_curve.copy()
        curve["predicted_class"] = model.predict(X_curve)
        curve["pass_probability"] = model.predict_proba(X_curve)[:, 1]
        st.line_chart(curve.set_index("study_hours")[["predicted_class", "pass_probability"]])
        st.caption(f"{name} studied {hours} hours -> predicted class {prediction}.")