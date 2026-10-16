from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / 'student_pass_model.joblib'

st.set_page_config(page_title='Student Performance Predictor', page_icon='🎓')
st.title('Student Performance Predictor')
st.write('Estimate the expected class from study hours.')

student_name = st.text_input('Student Name', placeholder='Juan Dela Cruz')
study_hours = st.number_input(
    'Study Hours', min_value=0.0, max_value=24.0, value=5.0, step=0.5
)
attendance = st.number_input(
    'Attendance Percentage', min_value=0.0, max_value=100.0, value=85.0, step=1.0,
    help='Collected for display. The saved model was trained using study hours only, so attendance does not change its prediction.',
)

if st.button('Predict Performance', type='primary'):
    if not student_name.strip():
        st.error('Enter the student name to continue.')
    elif not MODEL_PATH.exists():
        st.error(f'Model file not found: {MODEL_PATH.name}. Run train_model.py first.')
    else:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(pd.DataFrame({'study_hours': [study_hours]}))[0]
        label = 'Pass' if str(prediction) == '1' else 'Fail'

        st.subheader('Prediction result')
        st.write(f'**Student:** {student_name.strip()}')
        st.write(f'**Study Hours:** {study_hours:g}')
        st.write(f'**Attendance:** {attendance:g}%')
        st.success(f'**Prediction:** {label}')
        st.caption(
            'The app loaded the saved model and used study hours as its input feature. '
            'Attendance is displayed but does not affect this prediction because the model '
            'was trained without attendance data.'
        )

        chart_data = pd.DataFrame({
            'Study Hours': [study_hours],
            'Predicted Class': [label],
        })
        chart = (
            alt.Chart(chart_data)
            .mark_circle(size=220)
            .encode(
                x=alt.X('Study Hours:Q', scale=alt.Scale(domain=[0, 24]), title='Study Hours'),
                y=alt.Y(
                    'Predicted Class:N',
                    sort=['Fail', 'Pass'],
                    title='Predicted Class',
                    axis=alt.Axis(labelAngle=0),
                ),
                color=alt.Color(
                    'Predicted Class:N',
                    scale=alt.Scale(domain=['Fail', 'Pass'], range=['#d9534f', '#2e8b57']),
                    legend=None,
                ),
                tooltip=['Study Hours:Q', 'Predicted Class:N'],
            )
            .properties(title='Predicted class at the entered study hours', height=220)
        )
        st.subheader('Study hours and predicted class')
        st.altair_chart(chart, use_container_width=True)
