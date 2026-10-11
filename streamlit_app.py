"""Student Pass/Fail Predictor — Simple Green Dashboard."""

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

# -------------------- Green Theme --------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f9f4;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #14532d;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        background-color: #166534;
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 30px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #dcfce7;
        margin-bottom: 0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 12px;
        border-color: #d1e7d5;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Sidebar --------------------

with st.sidebar:
    st.title("🎓 StudentAI")
    st.caption("STUDENT PERFORMANCE SYSTEM")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Prediction History", "About Model"],
    )

    st.divider()

    if MODEL_PATH.exists():
        st.success("Model file found")
    else:
        st.error("Model file not found")

    st.caption("Machine Learning Project")

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
            <p>Predict student results using daily study hours.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.history

    total = len(history)
    passes = sum(
        item["Prediction"] == "Pass"
        for item in history
    )
    fails = total - passes
    pass_rate = passes / total * 100 if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total)
    col2.metric("Predicted Pass", passes)
    col3.metric("Predicted Fail", fails)
    col4.metric("Pass Rate", f"{pass_rate:.1f}%")

    st.subheader("📝 Make a Prediction")

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.write("### Student Information")

            with st.form("prediction_form"):
                student_name = st.text_input(
                    "Student Name",
                    placeholder="Enter student name",
                )

                study_hours = st.number_input(
                    "Daily Study Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=4.0,
                    step=0.5,
                )

                submitted = st.form_submit_button(
                    "Predict Result",
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
                                "Unexpected model output. "
                                "Expected 0 (Fail) or 1 (Pass)."
                            )
                        else:
                            result = "Pass" if prediction == 1 else "Fail"

                            record = {
                                "Student": student_name.strip(),
                                "Study Hours": float(study_hours),
                                "Prediction": result,
                            }

                            st.session_state.history.insert(0, record)
                            st.session_state.latest = record

                            st.rerun()

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

    with right:
        with st.container(border=True):
            st.write("### 📊 Latest Prediction")

            latest = st.session_state.get("latest")

            if latest:
                st.write(f"**Student:** {latest['Student']}")
                st.write(
                    f"**Daily Study Hours:** "
                    f"{latest['Study Hours']:.1f} hours"
                )

                if latest["Prediction"] == "Pass":
                    st.success("PASS — Predicted to pass")
                else:
                    st.error("FAIL — Predicted to fail")

            else:
                st.info("Submit a prediction to see the result here.")

    st.divider()

    st.subheader("📈 Study Hours Analysis")

    hours_range = [x / 2 for x in range(49)]

    chart_df = pd.DataFrame({
        "Study Hours": hours_range,
        "Predicted Class": [
            int(model.predict([[hours]])[0])
            for hours in hours_range
        ],
    })

    chart_df["Predicted Result"] = chart_df["Predicted Class"].map({
        0: "Fail",
        1: "Pass",
    })

    st.line_chart(
        chart_df.set_index("Study Hours")[["Predicted Class"]]
    )

    st.caption(
        "0 = Fail | 1 = Pass. Predictions are educational estimates "
        "based on the trained model, not guaranteed academic results."
    )

# -------------------- Prediction History --------------------

elif page == "Prediction History":

    st.title("📋 Prediction History")
    st.write("View the predictions made during this session.")

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
        st.info("No predictions yet. Visit the Dashboard to get started.")

# -------------------- About Model --------------------

elif page == "About Model":

    st.title("ℹ️ About the Model")

    st.write(
        "This application uses a trained machine-learning model "
        "to predict student pass/fail results based on daily study hours."
    )

    st.subheader("How It Works")

    st.markdown(
        """
        1. **Train:** `train_model.py` trains the model.
        2. **Save:** `joblib.dump()` saves the trained model.
        3. **Load:** `joblib.load()` loads the saved model.
        4. **Predict:** `model.predict()` predicts the result.
        5. **Display:** The application shows Pass or Fail.
        """
    )

    st.subheader("Model Information")

    st.write("**Model File:** `student_pass_model.joblib`")
    st.write("**Input:** Average daily study hours")
    st.write("**Output:** 0 = Fail, 1 = Pass")

    st.warning(
        "This project is for educational purposes. "
        "Its predictions depend on the training data and do not "
        "guarantee a student's actual academic performance."
    )
