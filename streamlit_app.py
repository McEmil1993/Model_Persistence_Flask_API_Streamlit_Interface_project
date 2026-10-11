"""Streamlit interface for the student pass-prediction model."""

import streamlit as st

from train_model import load_model, pass_probability, prediction_label


st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #e0f2fe 0%, #ede9fe 50%, #fdf2f8 100%);
        }
        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.72);
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎓 Student Pass Predictor")
st.write("Estimate whether a student will pass based on study hours and attendance.")

model = load_model()
name = st.text_input("Student name", placeholder="e.g., Alex Cruz")
study_hours = st.number_input("Study hours", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
attendance_percentage = st.number_input(
    "Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0
)

if st.button("Predict result", type="primary"):
    student_name = name.strip() or "Student"
    result = prediction_label(model, study_hours, attendance_percentage)
    probability = pass_probability(model, study_hours, attendance_percentage) * 100

    if result == "Pass":
        st.success(f"{student_name} is predicted to **Pass**.")
    else:
        st.warning(f"{student_name} is predicted to **Fail**.")

    st.write(
        f"The model used **{study_hours:g} study hours** and "
        f"**{attendance_percentage:g}% attendance**, and estimates a "
        f"{probability:.1f}% probability of passing. Attendance adjusts the effective "
        "study-hours score around a 75% reference point."
    )

    st.subheader("Pass probability by study hours")
    st.caption(
        f"Study hours: {study_hours:g} | Attendance: {attendance_percentage:g}% | "
        f"Predicted class: {result}"
    )
    st.metric("Predicted pass probability", f"{probability:.1f}%")

    # Draw the chart as SVG so Streamlit doesn't need pandas/NumPy to plot it.
    chart_limit = max(10.0, study_hours + 1.0)
    chart_width, chart_height = 640, 320
    left, right, top, bottom = 64, 20, 24, 48
    plot_width = chart_width - left - right
    plot_height = chart_height - top - bottom
    chart_hours = [step * chart_limit / 100 for step in range(101)]
    points = " ".join(
        f"{left + hours / chart_limit * plot_width:.1f},"
        f"{top + (1 - pass_probability(model, hours, attendance_percentage)) * plot_height:.1f}"
        for hours in chart_hours
    )
    selected_x = left + study_hours / chart_limit * plot_width
    selected_y = top + (1 - probability) * plot_height
    grid = "".join(
        f'<line x1="{left}" y1="{top + fraction * plot_height:.1f}" '
        f'x2="{chart_width - right}" y2="{top + fraction * plot_height:.1f}" '
        'stroke="#dbe3ef" stroke-width="1" />'
        f'<text x="{left - 10}" y="{top + fraction * plot_height + 4:.1f}" '
        'text-anchor="end" fill="#475569" font-size="12">'
        f'{int((1 - fraction) * 100)}%</text>'
        for fraction in (0, 0.25, 0.5, 0.75, 1)
    )
    st.markdown(
        f'''<svg viewBox="0 0 {chart_width} {chart_height}" role="img"
        aria-label="Estimated pass probability by study hours at the selected attendance, with the selected value highlighted"
        style="width:100%;height:auto;font-family:sans-serif">
        {grid}
        <polyline points="{points}" fill="none" stroke="#4f46e5"
        stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
        <line x1="{selected_x:.1f}" y1="{top}" x2="{selected_x:.1f}"
        y2="{chart_height - bottom}" stroke="#ec4899" stroke-dasharray="5 4" />
        <circle cx="{selected_x:.1f}" cy="{selected_y:.1f}" r="6"
        fill="#ec4899" stroke="white" stroke-width="2" />
        <text x="{left + plot_width / 2:.1f}" y="{chart_height - 8}"
        text-anchor="middle" fill="#475569" font-size="13">Study hours</text>
        <text x="{selected_x:.1f}" y="{max(top + 14, selected_y - 12):.1f}"
        text-anchor="middle" fill="#be185d" font-size="12">You: {study_hours:g}h</text>
        </svg>''',
        unsafe_allow_html=True,
    )