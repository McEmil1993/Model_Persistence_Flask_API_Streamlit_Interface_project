# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

st.title("🎓 Student Pass Predictor")
st.write("Enter the student's details below to predict whether they will pass.")

st.divider()

student_name = st.text_input("👤 Student Name", placeholder="e.g., Juan Dela Cruz")
study_hours = st.slider("📚 Study Hours per Day", min_value=0.0, max_value=12.0,
                        value=5.0, step=0.5)

if st.button("🔮 Predict Result", use_container_width=True):

    if not student_name.strip():
        st.warning("⚠️ Please enter the student's name.")
    else:
        prediction = int(model.predict([[study_hours]])[0])
        probability = float(model.predict_proba([[study_hours]])[0][1])

        st.divider()

        if prediction == 1:
            st.success(f"✅ **{student_name}** is likely to **PASS**!")
        else:
            st.error(f"❌ **{student_name}** is likely to **FAIL**.")

        col1, col2 = st.columns(2)
        col1.metric("Pass Probability", f"{probability:.2%}")
        col2.metric("Study Hours", f"{study_hours} hrs")

        with st.expander("ℹ️ How was this prediction generated?"):
            st.markdown(f"""
            The prediction was produced by a **Logistic Regression** model
            trained on historical study hours and pass/fail outcomes.

            1. The model applies: `z = ({model.coef_[0][0]:.3f} × study_hours) + ({model.intercept_[0]:.3f})`
            2. Converts `z` to probability via the sigmoid function.
            3. If probability ≥ 0.5 → **PASS**, else → **FAIL**.

            **For {student_name}:**  
            - Study hours = `{study_hours}`  
            - Pass probability = `{probability:.2%}`
            """)

        st.subheader("📊 Study Hours vs. Pass Probability")

        hours_range = np.linspace(0, 12, 100).reshape(-1, 1)
        probs = model.predict_proba(hours_range)[:, 1]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hours_range, probs, color="#1f77b4", linewidth=2,
                label="Pass Probability Curve")
        ax.axhline(0.5, color="gray", linestyle="--", label="Decision Threshold (0.5)")
        ax.scatter([study_hours], [probability], color="red", s=120,
                   zorder=5, label=f"{student_name or 'Student'} ({study_hours} hrs)")
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Probability of Passing")
        ax.set_title("Predicted Pass Probability vs. Study Hours")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

st.divider()
st.caption("Built with ❤️ using Streamlit + scikit-learn")