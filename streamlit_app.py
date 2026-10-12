import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Predictor")

name = st.text_input("Student name")
hours = st.number_input("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

if st.button("Predict"):
    X = pd.DataFrame({"study_hours": [hours]})
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0][1]
    who = name if name else "The student"

    if pred == 1:
        st.success(f"{who} is predicted to PASS ({prob:.0%} chance).")
    else:
        st.error(f"{who} is predicted to FAIL ({prob:.0%} chance of passing).")

    st.write(
        "The app sends the study hours to a logistic regression model that was "
        "trained earlier on past student data. The model turns the hours into a "
        "probability of passing. If it is 50% or higher, the prediction is Pass; "
        "otherwise Fail."
    )

    # Bonus chart: pass probability across study hours, with this student marked
    xs = np.linspace(0, max(12, hours + 2), 100)
    probs = model.predict_proba(pd.DataFrame({"study_hours": xs}))[:, 1]

    fig, ax = plt.subplots()
    ax.plot(xs, probs, label="Probability of passing")
    ax.axhline(0.5, linestyle="--", color="gray", label="Pass threshold")
    ax.scatter([hours], [prob], color="red", zorder=5, label=f"{who} ({hours} h)")
    ax.set_xlabel("Study hours")
    ax.set_ylabel("Probability of passing")
    ax.legend()
    st.pyplot(fig)