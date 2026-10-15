"""Streamlit UI — Exercise 3."""
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model.joblib")
FEATURES = ["study_hours", "attendance_percentage"]

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")
st.write("Enter student details, click **Predict Result**, and see the outcome with visual context.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --- 1. User inputs ---
student_name = st.text_input("Student Name", value="Alex")
study_hours = st.slider("Study Hours (per day)", min_value=0.0, max_value=12.0, value=6.5, step=0.5)
attendance = st.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=85, step=1)

# --- 2. Prediction action ---
if st.button("Predict Result"):
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Model file `student_pass_model.joblib` not found. Run `python train_model.py` first.")
        st.stop()

    X_new = pd.DataFrame([[study_hours, float(attendance)]], columns=FEATURES)
    pred = int(model.predict(X_new)[0])
    proba = float(model.predict_proba(X_new)[0][1])
    label = "Pass ✅" if pred == 1 else "Fail ❌"
    coef_hours, coef_att = model.coef_[0]
    intercept = model.intercept_[0]

    # Customized result message
    if pred == 1:
        st.success(f"🎉 **{student_name or 'Student'} is predicted to PASS** "
                   f"(confidence {proba:.1%}) with {study_hours} hrs/day and {attendance}% attendance!")
    else:
        st.error(f"📚 **{student_name or 'Student'} is predicted to FAIL** "
                 f"(P(pass)={proba:.1%}) with {study_hours} hrs/day and {attendance}% attendance. "
                 f"More study time / attendance may help!")

    # Explanation referencing weights/thresholds
    with st.expander("ℹ️ How was this prediction generated?", expanded=True):
        st.markdown(
            f"A **Logistic Regression** model trained on `{FEATURES}` computes a weighted sum:\n\n"
            f"`z = ({coef_hours:.3f} × study_hours) + ({coef_att:.4f} × attendance) + ({intercept:.3f})`\n\n"
            f"For this input: `z = ({coef_hours:.3f} × {study_hours}) + ({coef_att:.4f} × {attendance}) "
            f"+ ({intercept:.3f}) = {coef_hours * study_hours + coef_att * attendance + intercept:.3f}`.\n\n"
            f"Pass probability = `1 / (1 + e^(-z))` = **{proba:.1%}**. "
            f"Threshold is **50%**: ≥ 50% → Pass, < 50% → Fail.\n\n"
            f"Both weights are positive, so **more study hours and higher attendance each push the "
            f"prediction toward Pass**. Rough historical boundary: ~4.5+ hrs/day with ~70%+ attendance "
            f"tends to pass (see chart below)."
        )

    # --- 3. Visual chart (bonus): input vs typical pass/fail thresholds ---
    st.subheader("📊 Input vs Typical Pass Thresholds")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    categories = ["Study Hours\n(you vs 4.5h bar)", "Attendance %\n(you vs 70% bar)"]
    user_vals = [study_hours, attendance]
    threshold_vals = [4.5, 70]
    x = range(len(categories))
    ax.bar([i - 0.2 for i in x], user_vals, width=0.4, label="Your input")
    ax.bar([i + 0.2 for i in x], threshold_vals, width=0.4, label="Typical pass threshold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title(f"{student_name or 'Student'}: inputs vs pass thresholds")
    st.pyplot(fig)

    # Tabular comparison + native bar chart
    st.bar_chart(pd.DataFrame(
        {"Your input": user_vals, "Pass threshold": threshold_vals},
        index=["Study hours", "Attendance %"],
    ))
else:
    st.info("👆 Adjust the inputs above, then click **Predict Result**.")
    st.caption("Run with: `streamlit run streamlit_app.py`")
