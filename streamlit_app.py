# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────
model = joblib.load('student_pass_model.joblib')

# ─────────────────────────────────────────
# 2. PAGE CONFIG & TITLE
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Pass Predictor")
st.write("Enter the student details below to predict if they will pass or fail.")
st.markdown("---")

# ─────────────────────────────────────────
# 3. INPUT FIELDS
# ─────────────────────────────────────────
st.subheader("📝 Student Information")

student_name = st.text_input(
    "Student Name",
    placeholder="e.g., Juan Dela Cruz"
)

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

with col2:
    attendance = st.number_input(
        "Attendance Percentage (%)",
        min_value=0,
        max_value=100,
        value=85,
        step=1
    )

st.markdown("---")

# ─────────────────────────────────────────
# 4. PREDICTION BUTTON
# ─────────────────────────────────────────
if st.button("🔮 Predict", use_container_width=True):

    if not student_name.strip():
        st.warning("⚠️ Please enter the student's name.")
    else:
        # ─── Make prediction ───
        prediction = model.predict([[study_hours, attendance]])[0]
        result = "Pass" if prediction == 1 else "Fail"

        # ─── Show result ───
        st.markdown("### 📢 Prediction Result")

        if result == "Pass":
            st.success(f"✅ **{student_name}** is predicted to **PASS**!")
        else:
            st.error(f"❌ **{student_name}** is predicted to **FAIL**.")

        # ─── Summary metrics ───
        st.markdown("### 📊 Input Summary")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Student", student_name)
        with col_b:
            st.metric("Study Hours", f"{study_hours} hrs")
        with col_c:
            st.metric("Attendance", f"{attendance}%")

        # ─── Explanation ───
        with st.expander("ℹ️ How was this prediction generated?"):
            st.write(f"""
            **Model:** Decision Tree Classifier (loaded from `student_pass_model.joblib`)

            **Features used:**
            - Study Hours: **{study_hours}**
            - Attendance: **{attendance}%**

            **Process:**
            1. The saved model was loaded (no retraining needed).
            2. Your inputs were passed to `model.predict([[{study_hours}, {attendance}]])`.
            3. The model returned class **{prediction}** ({result}).

            **Legend:**
            - `0` = Fail ❌
            - `1` = Pass ✅
            """)

        st.markdown("---")

        # ─────────────────────────────────────────
        # 5. BONUS: CHART — Study Hours vs Predicted Class
        # ─────────────────────────────────────────
        st.subheader("📈 Study Hours vs Predicted Class")
        st.caption(f"Showing predictions for different study hours (attendance fixed at {attendance}%)")

        # Gumawa ng range ng study hours (0 to 12, step 0.5)
        hours_range = np.arange(0, 12.5, 0.5)

        # I-predict lahat ng values
        predictions = []
        for h in hours_range:
            pred = model.predict([[h, attendance]])[0]
            predictions.append(pred)

        # DataFrame para sa chart
        chart_data = pd.DataFrame({
            "Study Hours": hours_range,
            "Predicted Class": predictions
        }).set_index("Study Hours")

        # Line chart
        st.line_chart(chart_data)

        # Explanation ng chart
        st.markdown("##### 📖 Paano Basahin ang Chart?")
        st.write("""
        - **X-axis:** Study Hours (0 to 12)
        - **Y-axis:** Predicted Class
          - **0** = Fail ❌
          - **1** = Pass ✅
        - Ang **step** (pagbabago mula 0 to 1) ay ang **cutoff** ng model.
        - Sa chart na ito, ang **attendance** ay fixed sa value na na-input mo.
        """)

        st.markdown("---")

        # ─────────────────────────────────────────
        # 6. BONUS: COMPARE 2 ATTENDANCE LEVELS
        # ─────────────────────────────────────────
        st.subheader("📊 Effect of Attendance on Prediction")
        st.caption("Comparing LOW attendance (40%) vs HIGH attendance (90%)")

        # I-predict para sa 2 attendance values
        pred_low_att = [model.predict([[h, 40]])[0] for h in hours_range]
        pred_high_att = [model.predict([[h, 90]])[0] for h in hours_range]

        # DataFrame para sa comparison
        compare_data = pd.DataFrame({
            "Study Hours": hours_range,
            "Low Attendance (40%)": pred_low_att,
            "High Attendance (90%)": pred_high_att
        }).set_index("Study Hours")

        st.line_chart(compare_data)

        st.info("""
        💡 **Insight:** Kung mas mataas ang attendance, mas maagang nag-Pass ang model.
        Ibig sabihin, **nakakaapekto ang attendance** sa prediction.
        """)

# ─────────────────────────────────────────
# 7. FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.caption("🎓 IT Elective 3 — AI: Machine Learning | Model Persistence & Deployment")