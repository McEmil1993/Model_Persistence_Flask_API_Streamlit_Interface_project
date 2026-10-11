import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor")


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


model = load_model()

st.title("Student Pass Predictor")
st.write("Enter a student's name and study hours to predict Pass or Fail.")

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        features = pd.DataFrame({"study_hours": [hours]})
        result = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0][1])

        if result == 1:
            st.success(f"{name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to FAIL (pass probability {prob:.0%}).")

        st.subheader("How was this predicted?")
        st.write(
            "The model was trained earlier on past student data and saved with joblib. "
            "It looks at the study hours you entered and calculates a pass probability. "
            "If the probability is 50% or higher, the prediction is Pass; otherwise Fail. "
            "The model is only loaded here, not retrained."
        )

        # Bonus: chart of study hours vs. predicted class
        hours_range = [h / 2 for h in range(0, 49)]  # 0 to 24 in 0.5 steps
        curve = pd.DataFrame({"study_hours": hours_range})
        curve["pass_probability"] = model.predict_proba(curve[["study_hours"]])[:, 1]
        curve["predicted_class"] = model.predict(curve[["study_hours"]])

        st.subheader("Study hours vs. predicted class (1 = Pass, 0 = Fail)")
        st.line_chart(curve.set_index("study_hours")[["pass_probability", "predicted_class"]])
        st.caption(f"{name}'s input: {hours} hours")