import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

# ---- Small custom styling ----
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 28px 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: white; }
    .hero p  { margin: 6px 0 0; opacity: 0.9; }
    div.stButton > button {
        width: 100%;
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    div.stButton > button:hover { background: #4338ca; color: white; }
    </style>
    <div class="hero">
        <h1>🎓 Student Pass Predictor</h1>
        <p>Enter study hours (and attendance, if you have it) to predict pass or fail.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Two models: one uses study hours only, one uses study hours + attendance
model_hours = joblib.load("student_pass_model.joblib")
model_both = joblib.load("student_pass_model_attendance.joblib")

# ---- Inputs ----
name = st.text_input("👤 Student name")

col1, col2 = st.columns(2)
with col1:
    hours = st.number_input("📚 Study hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
with col2:
    attendance_text = st.text_input("🗓️ Attendance % (optional)", value="", placeholder="0 - 100")

# ---- Prediction button ----
if st.button("🔮 Predict"):
    attendance = None
    error = None

    if not name.strip():
        error = "Please enter the student's name."

    # Attendance is optional: blank means "not provided"
    if error is None and attendance_text.strip() != "":
        try:
            attendance = float(attendance_text)
        except ValueError:
            error = "Attendance must be a number."
        else:
            if not 0 <= attendance <= 100:
                error = "Attendance must be between 0 and 100."

    if error:
        st.warning(error)
    else:
        # Choose the model depending on whether attendance was given
        if attendance is None:
            X = pd.DataFrame({"study_hours": [hours]})
            model = model_hours
            used = "study hours only"
        else:
            X = pd.DataFrame({"study_hours": [hours], "attendance_percentage": [attendance]})
            model = model_both
            used = "study hours and attendance"

        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        st.divider()

        # ---- Result message ----
        if pred == 1:
            st.success(f"✅ {name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"❌ {name} is predicted to FAIL (probability of passing {prob:.0%}).")

        # ---- Summary cards ----
        m1, m2, m3 = st.columns(3)
        m1.metric("Study hours", f"{hours:g}")
        m2.metric("Attendance", f"{attendance:g}%" if attendance is not None else "Not given")
        m3.metric("Chance of passing", f"{prob:.0%}")
        st.progress(prob)

        # ---- Short explanation ----
        st.info(
            f"The model was trained on past students' results and used {used} for this prediction. "
            f"It calculated a {prob:.0%} chance of passing. "
            f"If that is 50% or higher it predicts PASS, otherwise FAIL."
        )

        # ---- Bonus chart: probability curve + this student's point ----
        xs = np.linspace(0, 12, 100)
        if attendance is None:
            curve_X = pd.DataFrame({"study_hours": xs})
            title = "Study hours only"
        else:
            curve_X = pd.DataFrame({
                "study_hours": xs,
                "attendance_percentage": [attendance] * len(xs),
            })
            title = f"Study hours (attendance fixed at {attendance:g}%)"
        probs = model.predict_proba(curve_X)[:, 1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, probs, color="#4f46e5", linewidth=2.5, label="Chance of passing")
        ax.fill_between(xs, probs, alpha=0.12, color="#4f46e5")
        ax.axhline(0.5, linestyle="--", color="gray", label="50% threshold")
        dot_color = "#16a34a" if pred == 1 else "#dc2626"
        ax.scatter([hours], [prob], color=dot_color, s=90, zorder=5, label=name)
        ax.set_xlabel("Study hours")
        ax.set_ylabel("Probability of passing")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
        st.pyplot(fig)

st.caption("Built with Streamlit and scikit-learn.")