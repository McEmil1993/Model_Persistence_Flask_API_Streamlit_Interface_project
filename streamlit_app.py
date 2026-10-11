# Exercise 3: Streamlit UI
import requests
import streamlit as st
import pandas as pd
import numpy as np

API_URL = 'http://127.0.0.1:8000/predict'
FEATURES = ['study_hours', 'attendance_percentage']


@st.cache_resource
def _session():
    return requests.Session()


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;600;700&display=swap');
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        .stApp { background: #0B0F14; color: #E5E7EB; font-family: 'Inter', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { background: #121923; border: 1px solid #1F2A37; border-radius: 4px; padding: 0.25rem; }
        .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.75rem; color: #8B98A9; border-radius: 4px; }
        .stTabs [aria-selected="true"] { background: #0B0F14; color: #22D3EE; }
        .sidebar .sidebar-content { background: #0B0F14; border-right: 1px solid #1F2A37; }
        .stat-tile { background: #121923; border: 1px solid #1F2A37; border-radius: 4px; padding: 0.75rem; }
        .stat-label { font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.7rem; color: #8B98A9; }
        .stat-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.25rem; color: #E5E7EB; }
        .hero-badge { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 2.5rem; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.5rem 1rem; border: 1px solid #1F2A37; border-radius: 4px; display: inline-block; }
        .hero-badge.pass { color: #A3E635; box-shadow: 0 0 12px rgba(163, 230, 53, 0.25); border-color: #A3E635; }
        .hero-badge.fail { color: #FB7185; box-shadow: 0 0 12px rgba(251, 113, 133, 0.25); border-color: #FB7185; }
        .api-status { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.05em; }
        .api-online { color: #A3E635; }
        .api-offline { color: #FB7185; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_online(timeout=2):
    try:
        s = _session()
        r = s.post(API_URL, json={'study_hours': 0.0, 'attendance_percentage': 70.0}, timeout=timeout)
        if r.status_code in (200, 400):
            return True
    except Exception:
        pass
    return False


def render_result(name_val, attend_val, study_h_val):
    res = st.session_state.get('result')
    if not res:
        st.write('AWAITING INPUT')
        return
    name = res.get('name', name_val)
    prob = float(res.get('prob', 0.0))
    is_pass = bool(res.get('is_pass', False))

    if is_pass:
        st.markdown('<div class="hero-badge pass">PASS</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hero-badge fail">FAIL</div>', unsafe_allow_html=True)

    st.markdown(f"<p style='margin-top:0.75rem;font-family:Inter,sans-serif;font-size:1rem;'>{name} is predicted to {'PASS' if is_pass else 'FAIL'} ({prob:.1%} probability)</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat-tile"><div class="stat-label">Probability</div><div class="stat-value">{:.1%}</div></div>'.format(prob), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-tile"><div class="stat-label">Study Hours</div><div class="stat-value">{}</div></div>'.format(float(study_h_val)), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-tile"><div class="stat-label">Attendance</div><div class="stat-value">{}%</div></div>'.format(int(float(attend_val))), unsafe_allow_html=True)


def render_chart(attendance_val):
    try:
        att = float(attendance_val)
        hours = np.arange(0, 13, 1)
        probs = []
        s = _session()
        for h in hours:
            try:
                r = s.post(API_URL, json={'study_hours': float(h), 'attendance_percentage': att}, timeout=5)
                if r.status_code == 200:
                    probs.append(float(r.json().get('pass_probability', np.nan)))
                else:
                    probs.append(np.nan)
            except Exception:
                probs.append(np.nan)
        dfc = pd.DataFrame({'study_hours': hours, 'pass_probability': probs})
        dfc_clean = dfc.dropna()
        if dfc_clean.empty:
            st.info('Chart unavailable (API not reachable).')
            return
        dfc_clean['threshold'] = 0.5
        st.line_chart(dfc_clean.set_index('study_hours'))
    except Exception as e:
        st.info(f'Chart unavailable: {e}')


def main():
    st.set_page_config(page_title='PassCheck', page_icon='🛰', layout='wide', initial_sidebar_state='expanded')
    inject_css()

    with st.sidebar:
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-weight:700;letter-spacing:0.15em;color:#E5E7EB;">PASSCHECK // v1</div>', unsafe_allow_html=True)
        st.divider()
        name = st.text_input('Student name')
        study_h = st.number_input('Study hours', min_value=0.0, max_value=24.0, step=0.5)
        attend = st.slider('Attendance percentage', min_value=0, max_value=100, value=70)
        st.divider()
        run = st.button('RUN PREDICTION', use_container_width=True)
        st.divider()
        if api_online():
            st.markdown('<div class="api-status api-online">● ONLINE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-status api-offline">● OFFLINE</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:1rem;"><h1 style="font-family:JetBrains Mono,monospace;font-weight:700;letter-spacing:0.1em;color:#E5E7EB;margin:0;">STUDENT OUTCOME PREDICTOR</h1><p style="margin:0.25rem 0 0 0;color:#8B98A9;font-size:0.9rem;">Enter study hours and attendance to predict whether a student will pass</p></div>', unsafe_allow_html=True)

    tab_res, tab_chart, tab_how = st.tabs(['Result', 'Chart', 'How it works'])

    with tab_res:
        if run:
            if not name.strip():
                st.warning('Please enter a student name.')
                st.session_state.pop('result', None)
            else:
                try:
                    payload = {'study_hours': float(study_h), 'attendance_percentage': float(attend)}
                    with st.spinner('Calling API...'):
                        s = _session()
                        resp = s.post(API_URL, json=payload, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        n = name.strip()
                        prob = float(data.get('pass_probability', 0.0))
                        is_pass = (data.get('result') == 'Pass')
                        st.session_state['result'] = {'name': n, 'prob': prob, 'is_pass': is_pass}
                    else:
                        try:
                            err = resp.json().get('error', resp.text)
                        except Exception:
                            err = resp.text
                        st.error(f'API error: {err}. Start the API first: python app.py')
                        st.session_state.pop('result', None)
                except requests.exceptions.ConnectionError:
                    st.error('Could not connect to Flask API at http://127.0.0.1:8000/predict. Start the API first: python app.py')
                    st.session_state.pop('result', None)
                except Exception as e:
                    st.error(f'Error calling API: {e}')
                    st.session_state.pop('result', None)
        render_result(name, attend, study_h)

    with tab_chart:
        render_chart(attend)

    with tab_how:
        st.write('Inputs sent to the API: study_hours and attendance_percentage (in order). A Logistic Regression pipeline (StandardScaler + LogisticRegression) was trained on data/students.csv. Pass when probability is at least 50%.')


if __name__ == '__main__':
    main()



