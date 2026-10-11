
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. LOAD THE EXISTING MODEL
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

# --------------------------------------------------
# 4. FOREST GREEN THEME
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F5F8F5;
        color: #24352B;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DFE8E0;
    }

    [data-testid="stSidebar"] h1 {
        color: #245C3A;
        font-size: 1.45rem;
    }

    h1, h2, h3 {
        color: #245C3A;
    }

    .main-title {
        color: #245C3A;
        font-size: 2.15rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #6B7D70;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DFE8E0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(36, 92, 58, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #617467;
    }

    [data-testid="stMetricValue"] {
        color: #245C3A;
        font-weight: 700;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #245C3A;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #19462B;
        color: white;
        border: none;
    }

    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #DFE8E0;
        border-radius: 14px;
        padding: 22px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    .section-note {
        color: #6B7D70;
        font-size: 0.9rem;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------

# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("# 🎓 StudentPass")
    st.caption("ACADEMIC PERFORMANCE SYSTEM")
    st.divider()

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    for item in ["Dashboard", "Predict Student", "About the Model"]:
        if st.button(
            item,
            key=f"nav_{item}",
            use_container_width=True,
            type="primary" if st.session_state.page == item else "secondary"
        ):
            st.session_state.page = item

    page = st.session_state.page

    st.divider()
    st.markdown("**Prediction Model**")
    st.caption("Logistic Regression")
    st.caption("Inputs: Study hours + attendance")
    st.caption("Output: Pass or Fail")


    st.divider()
    st.caption("Student Pass Prediction System")

# --------------------------------------------------
# 6. HELPER: PREDICT FROM STUDY HOURS
# --------------------------------------------------

def predict_result(study_hours, attendance=90.0):
    prediction = model.predict([[study_hours, attendance]])[0]
    return "Pass" if prediction == 1 else "Fail"

# --------------------------------------------------
# 7. DASHBOARD PAGE
# --------------------------------------------------
if page == "Dashboard":
    st.markdown(
        '<div class="main-title">Academic Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Overview of student pass predictions and study-hour patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history

    total = len(history)
    
    def get_prediction(item):
        value = item.get(
            "prediction",
            item.get("Prediction", "")
        )
        return str(value).strip().upper()

    passes = sum(
        1 for item in history
        if get_prediction(item) == "PASS"
    )

    fails = sum(
        1 for item in history
        if get_prediction(item) == "FAIL"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Predictions This Session", total)
    col2.metric("Predicted Pass", passes)
    col3.metric("Predicted Fail", fails)
    col4.metric(
    "Model Inputs",
    "2 Features",
    help="The model uses study hours and attendance percentage."
)
    st.write("")

    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Prediction by Study Hours")
        st.caption(
        "Predictions for study hours from 0 to 8, "
        "with attendance fixed at 90%."
        )
        hours = list(range(9))
        predictions = [predict_result(hour, 90.0) for hour in hours]

        chart_data = pd.DataFrame({
            "Study Hours": hours,
            "Predicted Result": predictions
        })

        chart_data["Class Value"] = chart_data[
            "Predicted Result"
        ].map({"Fail": 0, "Pass": 1})

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        ax.plot(
            chart_data["Study Hours"],
            chart_data["Class Value"],
            color="#2E7D4F",
            marker="o",
            linewidth=2.5,
            markersize=7
        )

        ax.set_xlabel("Study Hours per Day")
        ax.set_ylabel("Predicted Class")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fail", "Pass"])
        ax.set_xticks(hours)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DFE8E0")
        ax.spines["bottom"].set_color("#DFE8E0")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("How it works")

        st.markdown("""
**1. Enter student information**

Provide the student's name, daily study hours,
and attendance percentage.

**2. Generate a prediction**

The Logistic Regression model uses both input features
to predict Pass or Fail.

**3. Review the result**

View the predicted class and recent prediction history.
""")

        st.info(
            "This is an educational demonstration. "
            "Predictions depend on the sample data used to train "
            "the model and are not a guarantee of academic results."
        )

    st.subheader("Recent Predictions")

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(
            history_df.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption(
            "No predictions yet. Open Predict Student to get started."
        )

# --------------------------------------------------

elif page == "Predict Student":
    st.title("Predict Student Performance")
    st.write(
        "Enter the student's study hours and attendance "
        "to generate a prediction."
    )

    with st.form("prediction_form"):
        student_name = st.text_input("Student Name")

        study_hours = st.slider(
            "Study Hours per Day",
            min_value=0.0,
            max_value=12.0,
            value=5.0,
            step=0.5
        )

        attendance = st.slider(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0
        )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    
    if submitted:
        if not student_name.strip():
            st.warning("Please enter the student's name.")
        else:
            student_data = [[study_hours, attendance]]
            prediction = model.predict(student_data)[0]

            result = "PASS" if prediction == 1 else "FAIL"

            # Save the latest prediction
            st.session_state.latest_prediction = {
                "name": student_name.strip(),
                "study_hours": study_hours,
                "attendance": attendance,
                "prediction": result
            }

            # Save prediction to history
            st.session_state.prediction_history.insert(0, {
                "Student": student_name.strip(),
                "Study Hours": study_hours,
                "Attendance (%)": attendance,
                "Prediction": result
            })


    if st.session_state.latest_prediction:
        result_data = st.session_state.latest_prediction

        st.divider()
        st.subheader("Prediction Result")

        st.write(f"**Student:** {result_data['name']}")
        st.write(f"**Study Hours:** {result_data['study_hours']}")
        st.write(f"**Attendance:** {result_data['attendance']}%")

        if result_data["prediction"] == "PASS":
            st.success("Prediction: PASS")
        else:
            st.error("Prediction: FAIL")

        st.caption(
            "This is an educational prediction based on a "
            "demonstration model, not a guaranteed student outcome."
        )


# --------------------------------------------------
# 9. ABOUT THE MODEL PAGE
# --------------------------------------------------

elif page == "About the Model":
    st.title("Model Information")
    st.write(
        "Learn how the Student Pass Prediction System "
        "generates its predictions."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithm", "Logistic Regression")
    col2.metric("Input Features", "2")
    col3.metric("Output Classes", "2")

    st.divider()

    st.subheader("Input Features")

    st.markdown("""
    **1. Study hours per day**

    The number of hours a student spends studying each day.

    **2. Attendance percentage**

    The percentage of classes attended by the student.
    """)

    st.subheader("Possible Predictions")

    st.markdown("""
    - **PASS:** The model predicts that the student will pass.
    - **FAIL:** The model predicts that the student will fail.
    """)

    st.subheader("Model Persistence")

    st.markdown("""
    The trained model is stored in
    `student_pass_model.joblib`. The application loads the saved
    model using `joblib.load()`, so it does not need to retrain
    the model every time the interface opens.
    """)

    st.subheader("Important Limitations")

    st.warning(
        "This model was trained using manually created sample data "
        "for educational purposes. Its predictions and test accuracy "
        "have not been validated on real student records. The results "
        "should not be treated as reliable academic assessments."
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Student Pass Prediction System | Educational Project"
)
