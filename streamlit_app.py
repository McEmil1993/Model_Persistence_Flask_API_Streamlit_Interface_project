import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}

.main-title {
    background: linear-gradient(90deg, #17365D, #2874A6);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.main-title h1 {
    color: white;
}

.result-box {
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.pass-box {
    background-color: #D5F5E3;
    color: #196F3D;
}

.fail-box {
    background-color: #FADBD8;
    color: #922B21;
}

.footer {
    text-align: center;
    color: #777777;
    font-size: 13px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

try:
    model = load_model()
except Exception:
    st.error(
        "Model file not found. Please run train_model.py first "
        "and check that student_pass_model.joblib is in the "
        "same folder as this file."
    )
    st.stop()


st.markdown("""
<div class="main-title">
    <h1>🎓 Student Pass/Fail Predictor</h1>
    <p>AI Machine Learning Demonstration</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the student's information to generate a prediction "
    "using the trained machine learning model."
)

st.subheader("📝 Student Information")

with st.container(border=True):
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student's full name"
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    predict_button = st.button(
        "🔍 Predict Result",
        type="primary",
        use_container_width=True
    )


if predict_button:

    if not student_name.strip():
        st.warning("Please enter the student's name.")

    else:
        prediction_value = int(
            model.predict([[study_hours]])[0]
        )

        prediction = (
            "PASS" if prediction_value == 1 else "FAIL"
        )

        st.divider()
        st.subheader("📊 Prediction Result")

        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Student Name",
                student_name.strip()
            )

        with col2:
            st.metric(
                "Study Hours",
                f"{study_hours:g} hours"
            )

        
        if prediction == "PASS":
            st.markdown(
                '<div class="result-box pass-box">'
                '✅ PREDICTED RESULT: PASS'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box fail-box">'
                '⚠️ PREDICTED RESULT: FAIL'
                '</div>',
                unsafe_allow_html=True
            )

        st.write("")

        st.info(
            "Explanation: This prediction is based on the "
            "trained model and the study hours entered. "
            "It is an educational demonstration and does not "
            "guarantee the student's actual academic result."
        )

        
        st.divider()
        st.subheader("📈 Study Hours Analysis")

        st.write(
            "The chart compares the student's daily study "
            "hours with the remaining hours in a 24-hour day."
        )

        remaining_hours = 24 - study_hours

        fig1, ax1 = plt.subplots(figsize=(9, 3.5))

        ax1.barh(
            ["Daily Hours"],
            [study_hours],
            label="Study Hours",
            color="#2874A6"
        )

        ax1.barh(
            ["Daily Hours"],
            [remaining_hours],
            left=[study_hours],
            label="Remaining Hours",
            color="#D6EAF8"
        )

        ax1.set_xlim(0, 24)
        ax1.set_xlabel("Hours")
        ax1.set_title("Daily Study Hours Overview")
        ax1.legend(loc="lower center",
                   bbox_to_anchor=(0.5, -0.45),
                   ncol=2,
                   frameon=False)

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        
        st.subheader("🎯 Machine Learning Prediction Chart")

        st.write(
            "Class value 1 represents PASS, while class value "
            "0 represents FAIL. This value is a class label, "
            "not a probability or confidence percentage."
        )

        class_name = (
            "PASS (1)" if prediction_value == 1
            else "FAIL (0)"
        )

        fig2, ax2 = plt.subplots(figsize=(9, 3.5))

        bar_color = (
            "#239B56" if prediction_value == 1
            else "#CB4335"
        )

        ax2.barh(
            [class_name],
            [prediction_value],
            color=bar_color,
            height=0.5
        )

        ax2.set_xlim(0, 1.15)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["0 = FAIL", "1 = PASS"])
        ax2.set_title("Predicted Class")
        ax2.set_xlabel("Class Label")

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        
        st.subheader("📋 Prediction Summary")

        summary = pd.DataFrame({
            "Student Name": [student_name.strip()],
            "Study Hours": [study_hours],
            "Predicted Result": [prediction],
            "Class Value": [prediction_value]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

st.divider()

st.markdown("""
<div class="footer">
    IT Elective 3 – AI: Machine Learning<br>
    Student Pass/Fail Predictor | Educational Demonstration
</div>
""", unsafe_allow_html=True)