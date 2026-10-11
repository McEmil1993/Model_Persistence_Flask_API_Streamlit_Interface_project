
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Page settings
st.set_page_config(
    page_title="Student Pass Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Custom professional styling
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    h1, h2, h3 {
        color: #173b75;
    }
    div.stButton > button {
        background-color: #173b75;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.65rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #2455a0;
        color: white;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎓 Student Pass Prediction System")
st.caption("Predict a student outcome using study hours.")
st.divider()

# Student information section
with st.container(border=True):
    st.subheader("👤 Student Information")

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter the student's name"
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )

    predict_button = st.button("✨ Predict Result")

# Prediction section
if predict_button:
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[study_hours]])[0])

        st.divider()
        st.subheader("📋 Prediction Result")

        if prediction == 1:
            st.success(
                f"**{student_name.strip()}** — Predicted to Pass"
            )
        else:
            st.error(
                f"**{student_name.strip()}** — Predicted Not to Pass"
            )

        st.info(
            "The prediction was generated using the saved "
            "machine learning model and the study hours entered."
        )

        st.warning(
            "For educational purposes only. This sample model "
            "is not suitable for real academic decisions."
        )

        # Chart based on the model's training dataset
        st.divider()
        st.subheader("📊 Model Training Pattern")

        training_hours = list(range(1, 9))
        training_labels = [0, 0, 0, 0, 1, 1, 1, 1]

        fig, ax = plt.subplots(figsize=(9, 4.5))

        pass_hours = [
            h for h, label in zip(training_hours, training_labels)
            if label == 1
        ]
        fail_hours = [
            h for h, label in zip(training_hours, training_labels)
            if label == 0
        ]

        ax.scatter(
            fail_hours, [0] * len(fail_hours),
            color="#ef4444", s=85, label="Not pass (0)"
        )
        ax.scatter(
            pass_hours, [1] * len(pass_hours),
            color="#2563eb", s=85, label="Pass (1)"
        )

        ax.axvline(
            x=4.5,
            color="#64748b",
            linestyle="--",
            label="Training boundary (4.5 hours)"
        )

        ax.set_xlabel("Study Hours per Day")
        ax.set_ylabel("Training Label")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Not pass", "Pass"])
        ax.set_xticks(range(1, 9))
        ax.set_xlim(0.5, 8.5)
        ax.grid(axis="both", alpha=0.2)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
