import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="Student Pass Predictor", page_icon=None)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background-color: #0F172A;   /* deep navy */
    }

    /* Title styling */
    h1 {
        text-align: center;
        color: #F8FAFC;
        background-color: #1E293B;
        padding: 18px;
        border-radius: 12px;
        border: 2px solid #334155;
        margin-bottom: 25px;
    }

    /* Labels and text */
    label, .stMarkdown, p, .stText {
        color: #E2E8F0 !important;
    }

    /* Predict button styling */
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: #FFFFFF;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Student Pass Predictor")

# Load model
model = joblib.load("student_pass_model.joblib")

# Inputs
name = st.text_input("Student Name")
hours = st.number_input("Study Hours", min_value=0.0, max_value=40.0, value=5.0)

# Button
if st.button("Predict"):
    if not name:
        st.warning("Please enter a name.")
    else:
        result = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])

        if result == 1:
            st.success(f"{name} will PASS!")
        else:
            st.error(f"{name} will FAIL!")

        st.write(f"Pass probability: {prob:.1%}")

        st.info(f"""
        The model uses Logistic Regression. Your input of {hours} hours 
        gives a pass probability of {prob:.1%}. 
        If probability >= 50%, the result is PASS.
        """)

        st.subheader("Study Hours vs Pass Probability")

        hours_range = np.linspace(0, 15, 200)
        probs = model.predict_proba(hours_range.reshape(-1, 1))[:, 1]

        chart_data = pd.DataFrame({
            "Study Hours": hours_range,
            "Pass Probability": probs,
        })
        chart_data["Outcome"] = np.where(
            chart_data["Pass Probability"] >= 0.5, "Pass", "Fail"
        )

        student_point = pd.DataFrame({
            "Study Hours": [hours],
            "Pass Probability": [prob],
            "Outcome": ["Pass" if prob >= 0.5 else "Fail"],
        })

        # Deep, solid colors (no light pastel tones)
        color_scale = alt.Scale(
            domain=["Fail", "Pass"],
            range=["#DC2626", "#16A34A"]  # strong red, strong green
        )

        line = alt.Chart(chart_data).mark_line(size=3).encode(
            x=alt.X("Study Hours:Q", title="Study Hours"),
            y=alt.Y(
                "Pass Probability:Q",
                title="Pass Probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "Outcome:N",
                scale=color_scale,
                legend=alt.Legend(
                    title="Outcome",
                    orient="bottom",
                    direction="horizontal",
                    labelColor="#E2E8F0",
                    titleColor="#E2E8F0",
                ),
            ),
        )

        threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_line(
            strokeDash=[5, 5], color="#94A3B8"
        ).encode(y="y:Q")

        point = alt.Chart(student_point).mark_circle(
            size=200, stroke="#F8FAFC", strokeWidth=2
        ).encode(
            x="Study Hours:Q",
            y="Pass Probability:Q",
            color=alt.Color("Outcome:N", scale=color_scale, legend=None),
            tooltip=["Study Hours", "Pass Probability"],
        )

        chart = (line + threshold + point).properties(
            height=400,
            background="#1E293B",
        ).configure_axis(
            labelColor="#E2E8F0",
            titleColor="#E2E8F0",
            gridColor="#334155",
            domainColor="#475569",
        ).configure_view(
            strokeWidth=0,
        )

        st.altair_chart(chart, use_container_width=True)