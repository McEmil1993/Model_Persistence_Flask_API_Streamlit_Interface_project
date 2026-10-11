
import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()


# Custom CSS
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f4f7fc;
    color: #000000;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
    color: #000000;
}

/* Headings and paragraphs */
h1, h2, h3, h4, h5, h6, p, label {
    color: #000000 !important;
}

/* General text */
[data-testid="stMarkdownContainer"] {
    color: #000000 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e8eef8;
}

[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Input fields */
input {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Input labels */
[data-testid="stWidgetLabel"] * {
    color: #000000 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
}

/* Metric labels and values */
[data-testid="stMetric"] * {
    color: #000000 !important;
}

/* Prediction button */
.stButton > button {
    background-color: #2563eb;
    color: #000000 !important;
    border-radius: 10px;
    padding: 10px 25px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: #000000 !important;
}

/* Prediction messages */
[data-testid="stAlert"] * {
    color: #000000 !important;
}

/* Result cards */
.result-card {
    background: #ffffff;
    color: #000000;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    margin-top: 15px;
}

/* Captions */
[data-testid="stCaptionContainer"] * {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("🎓 Student Predictor")
    st.markdown("---")
    st.subheader("Navigation")
    st.write("📊 Dashboard")
    st.write("📈 Performance Analysis")
    st.write("ℹ️ About the Model")
    st.markdown("---")
    st.caption("Machine Learning Project")

# Header
st.title("Student Performance Dashboard")
st.write(
    "Predict student performance using a trained "
    "machine learning model."
)

st.divider()

# Student information
st.subheader("Student Information")

col1, col2 = st.columns(2)

with col1:
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name"
    )

with col2:
    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

predict_button = st.button(
    "Generate Prediction",
    use_container_width=True
)

# Prediction
if predict_button:

    if not student_name.strip():
        st.warning("Please enter a student name.")

    else:
        prediction = int(
            model.predict([[study_hours]])[0]
        )

        probabilities = model.predict_proba(
            [[study_hours]]
        )[0]

        pass_index = list(model.classes_).index(1)
        pass_probability = float(
            probabilities[pass_index] * 100
        )

        result = "PASS" if prediction == 1 else "FAIL"

        st.divider()
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Student", student_name)

        with col2:
            st.metric("Study Hours", f"{study_hours} hrs")

        with col3:
            st.metric("Pass Probability",
                      f"{pass_probability:.1f}%")

        if prediction == 1:
            st.success(
                f"Predicted Result: {result}"
            )
        else:
            st.error(
                f"Predicted Result: {result}"
            )

        # Curved line chart
        st.subheader("Performance Probability Curve")

        hours_range = np.linspace(0, 24, 200)

        pass_probabilities = model.predict_proba(
            hours_range.reshape(-1, 1)
        )[:, pass_index] * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hours_range,
            y=pass_probabilities,
            mode="lines",
            name="Pass Probability",
            line=dict(
                color="#2563eb",
                width=4,
                shape="spline",
                smoothing=0.8
            ),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.08)"
        ))

        # Student marker
        fig.add_trace(go.Scatter(
            x=[study_hours],
            y=[pass_probability],
            mode="markers",
            name=student_name,
            marker=dict(
                size=14,
                color="#f59e0b",
                line=dict(
                    color="white",
                    width=2
                )
            )
        ))

        # Decision threshold
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text="50% Threshold"
        )


        fig.update_layout(
            font=dict(
                color="#FFFFFF",
                family="Arial",
                size=14
            ),

            paper_bgcolor="#182848",
            plot_bgcolor="#182848",

            xaxis=dict(
                title="Study Hours",
                range=[0, 24],
                color="#FFFFFF",
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF"),
                gridcolor="rgba(255,255,255,0.15)"
            ),

            yaxis=dict(
                title="Pass Probability (%)",
                range=[0, 100],
                color="#FFFFFF",
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF"),
                gridcolor="rgba(255,255,255,0.15)"
            ),

            legend=dict(
                orientation="h",
                y=1.12,
                font=dict(color="#FFFFFF")
            ),

            hoverlabel=dict(
                bgcolor="#182848",
                font_color="#FFFFFF"
            ),

            height=420,
            hovermode="x unified",
            margin=dict(l=40, r=30, t=50, b=40)
        )



        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Explanation
        st.subheader("How Was This Predicted?")

        st.info(
            f"The machine learning model analyzed "
            f"{student_name}'s {study_hours} study hours. "
            f"Based on patterns learned during training, "
            f"it estimated a {pass_probability:.1f}% "
            f"probability of passing. "
            f"The predicted classification is {result}. "
            f"This is a model estimate, not a guaranteed "
            f"academic outcome."
        )

else:
    st.info(
        "Enter student information and click "
        "'Generate Prediction' to view the results."
    )

st.divider()
st.caption(
    "Student Performance Prediction System | "
    "Powered by Scikit-learn and Streamlit"
)
