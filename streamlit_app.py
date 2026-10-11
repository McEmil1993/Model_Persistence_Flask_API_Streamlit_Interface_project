
import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Professional blue-and-white design
st.markdown("""
<style>
    .stApp {
        background-color: #F5F8FC;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #17365D;
    }

    p, label {
        color: #344563;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background-color: #FFFFFF;
        border: 1px solid #D5E0EF;
        border-radius: 8px;
    }

    div.stButton > button {
        background-color: #1D4ED8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #173EA5;
        color: white;
        border: none;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #D8E2F0;
        border-radius: 8px;
        overflow: hidden;
    }

    hr {
        border-color: #D8E2F0;
    }

    .subtitle {
        color: #64748B;
        font-size: 0.95rem;
        margin-top: -8px;
    }

    .section-note {
        color: #64748B;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Store prediction history in the current session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Application heading
st.title("Student Performance Prediction")

st.markdown(
    '<p class="subtitle">'
    'Predict student results based on study hours.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

# Student inputs
st.subheader("Student Information")

student_name = st.text_input("Enter Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=0.5
)

# Prediction button and result
if st.button("Predict Result", use_container_width=True):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        # Make prediction using the saved model
        prediction = model.predict([[study_hours]])[0]
        result = "Pass" if prediction == 1 else "Fail"

        # Save this prediction to the session history
        st.session_state.prediction_history.append({
            "Student Name": student_name.strip(),
            "Study Hours": study_hours,
            "Predicted Result": result
        })

        # Save the latest result for display
        st.session_state.latest_prediction = {
            "Student Name": student_name.strip(),
            "Study Hours": study_hours,
            "Predicted Result": result
        }

# Display the latest prediction after each rerun
if "latest_prediction" in st.session_state:
    latest = st.session_state.latest_prediction

    st.subheader("Prediction Result")

    if latest["Predicted Result"] == "Pass":
        st.success(
            f'{latest["Student Name"]} is predicted to PASS!'
        )
    else:
        st.error(
            f'{latest["Student Name"]} is predicted to FAIL.'
        )

    st.write(f'**Student Name:** {latest["Student Name"]}')
    st.write(f'**Study Hours:** {latest["Study Hours"]}')
    st.write(f'**Prediction:** {latest["Predicted Result"]}')

    st.info(
        "This prediction is based on study hours and the "
        "trained machine learning model. It is only an estimate."
    )

# Bonus section
st.divider()
st.subheader("Bonus: Study Hours vs. Predicted Result")

st.markdown(
    '<p class="section-note">'
    'The table and chart update with each new prediction.'
    '</p>',
    unsafe_allow_html=True
)

# Show prediction history and chart only when predictions exist
if st.session_state.prediction_history:

    # Convert prediction history to a DataFrame
    chart_data = pd.DataFrame(
        st.session_state.prediction_history
    )

    # Display all predictions made during this session
    st.dataframe(
        chart_data,
        use_container_width=True,
        hide_index=True
    )

    # Prepare chart data
    chart_plot_data = pd.DataFrame({
        "Study Hours": chart_data["Study Hours"].astype(str),
        "Pass": [
            1 if result == "Pass" else 0
            for result in chart_data["Predicted Result"]
        ],
        "Fail": [
            1 if result == "Fail" else 0
            for result in chart_data["Predicted Result"]
        ]
    })

    # Use unique labels so repeated study-hour values
    # can still appear as separate predictions
    chart_plot_data["Prediction"] = [
        f"{i + 1}. {name} ({hours:g} hrs)"
        for i, (name, hours) in enumerate(
            zip(
                chart_data["Student Name"],
                chart_data["Study Hours"]
            )
        )
    ]

    chart_plot_data = chart_plot_data.set_index("Prediction")

    # Display chart
    st.subheader("Prediction Chart")

    st.caption(
        "Each bar represents a prediction. "
        "A value of 1 indicates the predicted class, "
        "while 0 indicates the other class."
    )

    st.bar_chart(
        chart_plot_data[["Pass", "Fail"]],
        color=["#16A34A", "#DC2626"]
    )

    st.caption("Green represents Pass; red represents Fail.")

else:
    st.info(
        "Enter a student's name and study hours, "
        "then click Predict Result to generate the chart."
    )

# Clear prediction history
if st.session_state.prediction_history:
    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []

        if "latest_prediction" in st.session_state:
            del st.session_state.latest_prediction

        st.rerun()
