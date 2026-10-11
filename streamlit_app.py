
import streamlit as st
import joblib
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Pass Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD SAVED MODEL ----------------
MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Model not found. Place student_pass_model.joblib "
        "in the same folder as streamlit_app.py."
    )
    st.stop()

# ---------------- SESSION DATA ----------------
if "records" not in st.session_state:
    st.session_state.records = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ---------------- BLUE AND WHITE THEME ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #F4F8FF;
        color: #172B4D;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF2FF;
        border-right: 1px solid #D5E4FA;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #174A8B;
    }

    h1, h2, h3 {
        color: #173F73;
    }

    .hero {
        background: linear-gradient(120deg, #174A8B, #3986E8);
        color: white;
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #EAF3FF;
        margin-bottom: 0;
    }

    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border: 1px solid #DCE8F8;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(23, 74, 139, 0.06);
        min-height: 130px;
    }

    .metric-label {
        color: #637793;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #1763B0;
        font-size: 30px;
        font-weight: 700;
    }

    .small-note {
        color: #71819A;
        font-size: 12px;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #DCE8F8;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #1769C2;
        color: white;
        border: 1px solid #1769C2;
        border-radius: 10px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #124E96;
        color: white;
        border-color: #124E96;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 22px;
        border: 1px solid #DCE8F8;
        border-radius: 16px;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border: 1px solid #DCE8F8;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HELPER FUNCTIONS ----------------
def predict_student(name, hours):
    prediction = int(model.predict([[float(hours)]])[0])
    result = "Pass" if prediction == 1 else "Fail"

    record = {
        "Student Name": name.strip(),
        "Study Hours": float(hours),
        "Prediction": result,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    st.session_state.records.append(record)
    st.session_state.last_prediction = record

    return result


def metric_card(label, value, description):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-note">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("# 🎓 StudentPass")
    st.caption("Student Performance System")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Student",
            "Analytics",
            "Prediction History"
        ]
    )

    st.divider()
    st.markdown("### Model Status")
    st.success("Saved model loaded")
    st.caption("Educational demonstration")


# ---------------- PREPARE SUMMARY ----------------
records_df = pd.DataFrame(st.session_state.records)

if records_df.empty:
    total = 0
    passes = 0
    fails = 0
    pass_rate = 0.0
else:
    total = len(records_df)
    passes = int((records_df["Prediction"] == "Pass").sum())
    fails = int((records_df["Prediction"] == "Fail").sum())
    pass_rate = passes / total * 100


# ================= DASHBOARD =================
if page == "Dashboard":
    show_hero(
        "Student Performance Dashboard",
        "Welcome! View student prediction summaries and quick results."
    )

    st.subheader("Dashboard Overview")
    st.caption("Summary of predictions made during this app session.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Predictions", total, "Students tested")

    with c2:
        metric_card("Predicted Pass", passes, "Pass results")

    with c3:
        metric_card("Predicted Fail", fails, "Fail results")

    with c4:
        metric_card("Pass Prediction Rate", f"{pass_rate:.1f}%", "Of tested students")

    st.write("")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Quick Prediction")

        with st.form("quick_prediction"):
            name = st.text_input(
                "Student Name",
                placeholder="Enter student name"
            )

            hours = st.number_input(
                "Study Hours per Day",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5
            )

            submit = st.form_submit_button(
                "Predict Result",
                use_container_width=True
            )

        if submit:
            if not name.strip():
                st.warning("Please enter the student's name.")
            else:
                result = predict_student(name, hours)

                if result == "Pass":
                    st.success(f"{name} — Predicted Pass")
                else:
                    st.error(f"{name} — Predicted Fail")

                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Recent Predictions")

        if records_df.empty:
            st.info("No predictions yet. Try the Quick Prediction form.")
        else:
            st.dataframe(
                records_df.tail(5).iloc[::-1],
                hide_index=True,
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "Prediction records are temporary and are not stored in a database."
    )


# ================= PREDICT STUDENT =================
elif page == "Predict Student":
    show_hero(
        "Predict Student Result",
        "Enter a student's details to test the saved model."
    )

    with st.form("prediction_form"):
        student_name = st.text_input(
            "Student Name",
            placeholder="e.g. Jenny"
        )

        study_hours = st.number_input(
            "Study Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

        submitted = st.form_submit_button(
            "✨ Predict Result",
            use_container_width=True
        )

    if submitted:
        if not student_name.strip():
            st.warning("Please enter the student's name.")
        else:
            result = predict_student(student_name, study_hours)

            st.subheader("Prediction Result")

            result_col, hours_col = st.columns(2)

            with result_col:
                if result == "Pass":
                    st.success(f"Prediction: {result}")
                else:
                    st.error(f"Prediction: {result}")

            with hours_col:
                st.metric("Study Hours per Day", f"{study_hours:.1f} hours")

            st.info(
                "The prediction uses study hours and sample training data. "
                "It is not a guarantee of actual academic performance."
            )


# ================= ANALYTICS =================
elif page == "Analytics":
    show_hero(
        "Student Analytics",
        "Explore the results and study-hour patterns of tested students."
    )

    if records_df.empty:
        st.info(
            "No data to display yet. Make predictions on the "
            "Predict Student page first."
        )
    else:
        a, b, c = st.columns(3)

        with a:
            st.metric("Total Students Tested", total)

        with b:
            st.metric("Predicted Pass", passes)

        with c:
            st.metric("Predicted Fail", fails)

        left, right = st.columns(2)

        with left:
            st.subheader("Prediction Distribution")

            distribution = (
                records_df["Prediction"]
                .value_counts()
                .reindex(["Pass", "Fail"], fill_value=0)
                .rename_axis("Result")
                .reset_index(name="Students")
            )

            result_chart = alt.Chart(distribution).mark_bar(
                cornerRadiusTopLeft=8,
                cornerRadiusTopRight=8
            ).encode(
                x=alt.X("Result:N", title="Prediction"),
                y=alt.Y("Students:Q", title="Number of Students"),
                color=alt.Color(
                    "Result:N",
                    scale=alt.Scale(
                        domain=["Pass", "Fail"],
                        range=["#2475D0", "#9CBCE8"]
                    ),
                    legend=None
                ),
                tooltip=["Result", "Students"]
            ).properties(height=320)

            st.altair_chart(
                result_chart,
                use_container_width=True
            )

        with right:
            st.subheader("Study Hours by Student")

            hours_chart = alt.Chart(records_df).mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6,
                color="#2475D0"
            ).encode(
                x=alt.X(
                    "Student Name:N",
                    title="Student",
                    axis=alt.Axis(labelAngle=-35)
                ),
                y=alt.Y(
                    "Study Hours:Q",
                    title="Study Hours",
                    scale=alt.Scale(domain=[0, 24])
                ),
                tooltip=[
                    "Student Name",
                    "Study Hours",
                    "Prediction"
                ]
            ).properties(height=320)

            st.altair_chart(
                hours_chart,
                use_container_width=True
            )

        st.subheader("All Prediction Records")
        st.dataframe(
            records_df,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download Results as CSV",
            data=records_df.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_history.csv",
            mime="text/csv"
        )


# ================= PREDICTION HISTORY =================
elif page == "Prediction History":
    show_hero(
        "Prediction History",
        "Search, review, and download results from this session."
    )

    if records_df.empty:
        st.info("No prediction history yet.")
    else:
        search_name = st.text_input(
            "Search Student Name",
            placeholder="Type a name to filter..."
        )

        filtered_df = records_df.copy()

        if search_name.strip():
            filtered_df = filtered_df[
                filtered_df["Student Name"].str.contains(
                    search_name.strip(),
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            filtered_df.iloc[::-1],
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download History as CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Current Session History"):
            st.session_state.records = []
            st.session_state.last_prediction = None
            st.rerun()


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Student Pass Prediction System | Blue & White Theme | "
    "Sample model for educational demonstration"
)
