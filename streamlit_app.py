import streamlit as st
import joblib
import pandas as pd

# ---------- Page setup ----------
st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #E6E6FA, #DA70D6);
        padding: 28px 30px;
        border-radius: 18px;
        color: #3b0764;
        margin-bottom: 22px;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: #3b0764; }
    .hero p { margin: 6px 0 0 0; opacity: 0.9; color: #3b0764; }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        padding: 0.7rem;
        font-weight: 600;
        border-radius: 10px;
    }
    div.stButton > button:hover { color: white; border: none; opacity: 0.9; }
    </style>
    <div class="hero">
        <h1>🎓 Student Pass Predictor</h1>
        <p>A saved machine learning model predicts if a student will pass or fail based on study hours.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Load the saved model (no retraining) ----------
model = joblib.load("student_pass_model.joblib")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("**Model:** Logistic Regression")
    st.write("**Input:** Study hours")
    st.write("**Output:** Pass (1) or Fail (0)")
    st.write("The model was trained once and loaded from `student_pass_model.joblib`.")

# ---------- Inputs ----------
if "history" not in st.session_state:
    st.session_state.history = []

with st.container(border=True):
    st.subheader("Student details")
    name = st.text_input("Student name", placeholder="Enter the student's name")
    hours = st.slider("Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    predict_clicked = st.button("Predict")

# ---------- Prediction ----------
if predict_clicked:
    if not name.strip():
        st.warning("Please enter the student name first.")
    else:
        result = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])
        label = "Pass" if result == 1 else "Fail"

        st.session_state.history.insert(
            0, {"Student": name.strip(), "Study hours": hours, "Pass chance": f"{prob:.0%}", "Result": label}
        )

        st.subheader("Result")
        if result == 1:
            st.success(f"🎉 {name.strip()} is predicted to **Pass**.")
        else:
            st.error(f"⚠️ {name.strip()} is predicted to **Fail**.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Study hours", f"{hours}")
        col2.metric("Pass chance", f"{prob:.0%}")
        col3.metric("Predicted class", f"{result} ({label})")
        st.progress(prob)

        # Short explanation
        st.info(
            f"**How the prediction was generated:** The saved model was trained on "
            f"study hours and pass/fail results. For {hours} study hours, it estimated "
            f"a {prob:.0%} chance of passing. Since this is "
            f"{'50% or higher' if result == 1 else 'below 50%'}, the predicted class is {label}."
        )

        # Bonus chart: study hours vs predicted class
        st.subheader("Study hours vs predicted class")
        hours_range = [h / 2 for h in range(0, 49)]
        chart_df = pd.DataFrame({
            "Study hours": hours_range,
            "Predicted class (1 = Pass, 0 = Fail)": [int(model.predict([[h]])[0]) for h in hours_range],
            "Pass chance": [float(model.predict_proba([[h]])[0][1]) for h in hours_range],
        }).set_index("Study hours")
        st.line_chart(chart_df)
        st.caption(f"{name.strip()} studied {hours} hours, predicted class: {result} ({label}).")

# ---------- History ----------
if st.session_state.history:
    st.subheader("Recent predictions")
    st.dataframe(pd.DataFrame(st.session_state.history).head(5), use_container_width=True, hide_index=True)