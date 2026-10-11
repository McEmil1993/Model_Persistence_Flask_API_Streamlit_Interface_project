import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")

FEATURES = ["study_hours", "attendance_percentage"]


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model_attendance.joblib")


model = load_model()

# ---------- Header ----------
st.title("Student Pass Predictor")
st.caption("Estimate a student's likelihood of passing based on weekly study hours and attendance.")
st.divider()

# ---------- Input card ----------
with st.container(border=True):
    st.subheader("Student details")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        name = col1.text_input("Student name", placeholder="e.g. Juan Dela Cruz")
        hours = col2.number_input(
            "Study hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5
        )
        attendance = col3.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0
        )
        submitted = st.form_submit_button("Run prediction", type="primary")

# ---------- Results ----------
if submitted:
    if not name.strip():
        st.warning("Please enter the student's name.")
    elif not 0 <= attendance <= 100:
        st.warning("Attendance must be between 0 and 100.")
    else:
        X = pd.DataFrame({"study_hours": [hours], "attendance_percentage": [attendance]})[FEATURES]
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        label = "PASS" if pred == 1 else "FAIL"

        st.subheader("Prediction result")

        if pred == 1:
            st.success(f"{name} is predicted to {label}.")
        else:
            st.error(f"{name} is predicted to {label}.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted outcome", label)
        m2.metric("Pass probability", f"{prob:.0%}")
        m3.metric("Study hours", f"{hours:g}")
        m4.metric("Attendance", f"{attendance:g}%")

        st.progress(prob, text=f"Pass probability: {prob:.0%}")

        with st.expander("How was this prediction generated?", expanded=True):
            st.write(
                "The model was trained once on past data (study hours and attendance "
                "percentage) and saved to a file. When you click **Run prediction**, the "
                "saved model loads, takes the two values you entered, and applies the "
                "pattern it learned to calculate a pass probability. If that probability "
                "is above 50%, the predicted class is **Pass**; otherwise it is **Fail**. "
                "No retraining happens."
            )

        # ---------- Chart ----------
        st.subheader("Study hours vs. predicted class")

        curve = pd.DataFrame({"study_hours": [x / 4 for x in range(0, 61)]})  # 0 to 15 hours
        curve["attendance_percentage"] = attendance
        curve["pass_prob"] = model.predict_proba(curve[FEATURES])[:, 1]

        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(curve["study_hours"], curve["pass_prob"], color="#4F46E5", linewidth=2.5,
                label=f"Model pass probability ({attendance:g}% attendance)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% threshold")
        ax.fill_between(curve["study_hours"], 0, 1,
                        where=curve["pass_prob"] >= 0.5, color="#22C55E", alpha=0.08)
        ax.fill_between(curve["study_hours"], 0, 1,
                        where=curve["pass_prob"] < 0.5, color="#EF4444", alpha=0.08)
        ax.scatter([hours], [prob], color="#EF4444" if pred == 0 else "#16A34A",
                   s=140, zorder=5, edgecolor="white", label=f"{name} ({label})")
        ax.set_xlabel("Study hours")
        ax.set_ylabel("Probability of passing")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="center right", frameon=False)
        st.pyplot(fig)
        st.caption("Green area = predicted Pass, red area = predicted Fail.")