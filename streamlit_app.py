
"""Modern Student Pass/Fail Predictor Dashboard."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Styling --------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    .hero {
        background: linear-gradient(120deg, #111827, #374151);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
        font-size: 32px;
    }

    .hero p {
        color: #d1d5db;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        margin: 10px 0 16px 0;
        color: #111827;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border-radius: 14px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 9px;
        min-height: 44px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Sidebar --------------------

with st.sidebar:
    st.markdown("# 🎓 StudentAI")
    st.caption("STUDENT PERFORMANCE SYSTEM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Prediction History", "About Model"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("### Model Status")

    if MODEL_PATH.exists():
        st.success("Model loaded")
        st.caption("student_pass_model.joblib")
    else:
        st.error("Model file not found")
        st.caption("Run: python train_model.py")

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("Educational use only")

# -------------------- Load Model --------------------

if not MODEL_PATH.exists():
    st.error(
        "The trained model is missing. Place "
        "`student_pass_model.joblib` in the same folder "
        "as `app.py`, or run `python train_model.py`."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Dashboard --------------------

if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>Student Performance Dashboard</h1>
            <p>Predict academic outcomes using average daily study hours.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(st.session_state.history)
    passes = sum(
        item["Prediction"] == "Pass"
        for item in st.session_state.history
    )
    fails = total - passes
    pass_rate = (passes / total * 100) if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total)
    col2.metric("Predicted Pass", passes)
    col3.metric("Predicted Fail", fails)
    col4.metric("Pass Rate", f"{pass_rate:.1f}%")

    st.markdown(
        '<div class="section-title">Make a Prediction</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("📝 Student Information")
            st.write("Enter the student's details below.")

            with st.form("prediction_form", clear_on_submit=False):
                student_name = st.text_input(
                    "Student name",
                    placeholder="Enter student name",
                )

                study_hours = st.number_input(
                    "Average daily study hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=4.0,
                    step=0.5,
                    help="Enter the average number of hours studied per day.",
                )

                st.caption(
                    "The model uses study hours as its prediction input."
                )

                submitted = st.form_submit_button(
                    "Predict Academic Result",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                if not student_name.strip():
                    st.warning("Please enter the student's name.")
                else:
                    try:
                        prediction = int(
                            model.predict([[float(study_hours)]])[0]
                        )

                        if prediction not in (0, 1):
                            st.error(
                                "The model returned an unexpected class. "
                                "Expected 0 (Fail) or 1 (Pass)."
                            )
                        else:
                            result = "Pass" if prediction == 1 else "Fail"

                            st.session_state.history.insert(
                                0,
                                {
                                    "Student": student_name.strip(),
                                    "Study Hours": float(study_hours),
                                    "Prediction": result,
                                },
                            )

                            st.session_state.latest = {
                                "Student": student_name.strip(),
                                "Study Hours": float(study_hours),
                                "Prediction": result,
                            }

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

    with right:
        with st.container(border=True):
            st.subheader("📊 Latest Prediction")

            latest = st.session_state.get("latest")

            if latest:
                st.caption("MOST RECENT RESULT")
                st.markdown(f"### {latest['Student']}")

                if latest["Prediction"] == "Pass":
                    st.success("PASS — Predicted to pass")
                else:
                    st.error("FAIL — Predicted to fail")

                st.metric(
                    "Daily Study Hours",
                    f"{latest['Study Hours']:.1f} hours",
                )

                st.progress(
                    min(latest["Study Hours"] / 24, 1.0),
                    text="Study hours relative to 24 hours",
                )

                st.caption(
                    "This progress bar represents study hours, "
                    "not the probability of passing."
                )
            else:
                st.info(
                    "Your prediction will appear here after you "
                    "submit the student information."
                )

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Study Hours Analysis</div>',
        unsafe_allow_html=True,
    )

    # Generate model predictions over a range of study hours.
    hours_range = [x / 2 for x in range(0, 49)]
    chart_df = pd.DataFrame({"Study Hours": hours_range})
    chart_df["Predicted Class"] = [
        int(model.predict([[hours]])[0])
        for hours in hours_range
    ]
    chart_df["Predicted Result"] = chart_df["Predicted Class"].map(
        {0: "Fail", 1: "Pass"}
    )

    chart_left, chart_right = st.columns([1.4, 0.8])

    with chart_left:
        with st.container(border=True):
            st.subheader("Prediction by Study Hours")
            st.caption("0 = Fail, 1 = Pass")

            st.line_chart(
                chart_df.set_index("Study Hours")[["Predicted Class"]],
                height=300,
            )

    with chart_right:
        with st.container(border=True):
            st.subheader("Model Output")
            st.write("Predictions across selected study-hour values.")

            st.dataframe(
                chart_df[
                    ["Study Hours", "Predicted Result"]
                ],
                use_container_width=True,
                hide_index=True,
                height=300,
            )

    st.caption(
        "Educational demonstration only. Predictions are based on study "
        "hours and do not guarantee actual academic performance."
    )

# -------------------- Prediction History --------------------

elif page == "Prediction History":
    st.title("📋 Prediction History")
    st.write("Review predictions made during this app session.")

    history = st.session_state.history

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download History as CSV",
            data=csv,
            file_name="student_prediction_history.csv",
            mime="text/csv",
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.session_state.pop("latest", None)
            st.rerun()
    else:
        st.info("No predictions yet. Visit Dashboard to get started.")

# -------------------- About Model --------------------

elif page == "About Model":
    st.title("ℹ️ About the Model")

    st.write(
        "This application uses a previously trained machine-learning "
        "model to predict a student's pass/fail outcome from daily "
        "study hours."
    )

    st.subheader("How It Works")

    st.markdown(
        """
        1. **Train:** `train_model.py` trains the machine-learning model.
        2. **Save:** `joblib.dump()` saves the trained model to disk.
        3. **Load:** `joblib.load()` loads the saved model into the app.
        4. **Predict:** The app passes the study hours to `model.predict()`.
        5. **Display:** The app shows the predicted class to the user.
        """
    )

    st.subheader("Model Information")

    st.write("**Model file:** `student_pass_model.joblib`")
    st.write("**Input feature:** Average daily study hours")
    st.write("**Output classes:** 0 = Fail, 1 = Pass")

    st.warning(
        "This is a learning project, not a validated academic assessment "
        "tool. Model quality depends on the training dataset. Do not "
        "interpret its output as a guaranteed result or probability."
    )
