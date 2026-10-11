from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_pass_model.joblib"

st.set_page_config(
    page_title="Student Success | Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
:root { --navy:#14213d; --blue:#3157d5; --muted:#667085; }
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f4f7fb;}
.block-container {padding-top:1.5rem; max-width:1450px;}
section[data-testid="stSidebar"] {background:#111c35;}
section[data-testid="stSidebar"] * {color:#eef4ff;}
.hero {padding:28px 32px; border-radius:22px; color:white;
background:linear-gradient(115deg,#14213d 0%,#2548b8 62%,#38bdf8 100%); margin-bottom:20px;}
.hero h1 {font-family:Manrope,sans-serif; font-size:2.1rem; font-weight:800; margin:4px 0 8px;}
.hero p {margin:0; color:#e0ecff; font-size:1rem;}
.eyebrow {font-size:.75rem; letter-spacing:.15em; text-transform:uppercase; font-weight:700; opacity:.9;}
.panel {background:white; padding:21px; border:1px solid #e5eaf3; border-radius:18px; box-shadow:0 7px 24px rgba(16,24,40,.045);}
.panel-title {font-weight:800; font-size:1.05rem; color:#172554; margin-bottom:3px;}
.muted {color:#667085; font-size:.9rem;}
.result-pass {background:#ecfdf5; border:1px solid #a7f3d0; border-radius:16px; padding:20px; color:#065f46;}
.result-fail {background:#fff1f2; border:1px solid #fecdd3; border-radius:16px; padding:20px; color:#9f1239;}
.result-label {font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.12em;}
.result-value {font-size:2.1rem; font-weight:800; line-height:1.25; margin:5px 0;}
div[data-testid="stMetric"] {background:#fff; border:1px solid #e5eaf3; padding:13px 16px; border-radius:14px;}
.stButton button {border-radius:11px; font-weight:700; min-height:2.7rem;}
div[data-testid="stTabs"] button {font-weight:700;}
footer {visibility:hidden;}
</style>
<div class="hero">
  <div class="eyebrow">Machine Learning · Student Analytics</div>
  <h1>Student Success Dashboard</h1>
  <p>Predict an individual student's outcome or review predictions for an entire class.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.warning("The trained model file is missing. Run `python train_model.py` in the project folder, then refresh this page.")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))


def predict_students(frame):
    """Predict outcomes for a dataframe containing Student and Study Hours."""
    hours = frame["Study Hours"].astype(float).to_numpy().reshape(-1, 1)
    predictions = model.predict(hours).astype(int)
    probabilities = model.predict_proba(hours)
    classes = list(model.classes_)
    pass_idx = classes.index(1) if 1 in classes else None
    pass_probability = probabilities[:, pass_idx] if pass_idx is not None else np.zeros(len(frame))
    output = frame.copy()
    output["Predicted Outcome"] = np.where(predictions == 1, "Pass", "Fail")
    output["Estimated Pass Probability"] = pass_probability
    output["Estimated Pass Probability (%)"] = (pass_probability * 100).round(1)
    return output


def save_last_prediction():
    """Add the most recent individual prediction to the class roster."""
    last = st.session_state.get("last_single_prediction")
    if last:
        current = st.session_state["class_predictions"]
        st.session_state["class_predictions"] = pd.concat(
            [current, pd.DataFrame([last])], ignore_index=True
        )


with st.sidebar:
    st.markdown("## 🎓 Student Success")
    st.caption("PREDICTION WORKSPACE")
    page = st.radio("Navigate", ["Dashboard", "Class predictions", "About the model"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model status**")
    st.success("Saved model loaded")
    st.caption("Input feature: study hours")
    st.caption("Output: predicted Pass / Fail")
    st.divider()
    st.caption("Demo project for learning model persistence and deployment.")

if "class_predictions" not in st.session_state:
    st.session_state["class_predictions"] = pd.DataFrame(
        columns=["Student", "Study Hours", "Predicted Outcome",
                 "Estimated Pass Probability", "Estimated Pass Probability (%)"]
    )

if page == "Dashboard":
    tab_one, tab_class = st.tabs(["👤 Individual student", "👥 Add students to class"])

    with tab_one:
        col_form, col_result = st.columns([0.85, 1.25], gap="large")
        with col_form:
            st.markdown('<div class="panel"><div class="panel-title">Student profile</div><div class="muted">Enter details to generate one prediction.</div></div>', unsafe_allow_html=True)
            with st.form("single_student_form", clear_on_submit=False):
                student_name = st.text_input("Student name", placeholder="e.g. Alex Santos")
                study_hours = st.slider("Study hours", min_value=0.0, max_value=12.0, value=4.0, step=0.5)
                st.caption("Study hours means the number of hours studied during the period represented by your training data.")
                submitted = st.form_submit_button("✨ Predict outcome", type="primary", use_container_width=True)

        with col_result:
            if submitted:
                clean_name = student_name.strip() or "Unnamed student"
                one = predict_students(pd.DataFrame({"Student": [clean_name], "Study Hours": [study_hours]})).iloc[0]
                outcome = one["Predicted Outcome"]
                probability = float(one["Estimated Pass Probability"])
                result_class = "result-pass" if outcome == "Pass" else "result-fail"
                st.markdown(
                    f'<div class="{result_class}"><div class="result-label">Model prediction for</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;margin-top:4px">{clean_name}</div>'
                    f'<div class="result-value">{outcome}</div>'
                    f'<div>{study_hours:g} study hours entered</div></div>',
                    unsafe_allow_html=True,
                )
                k1, k2 = st.columns(2)
                k1.metric("Estimated pass probability", f"{probability:.1%}")
                k2.metric("Study hours", f"{study_hours:g} hrs")
                # Show where this student's input falls on the model's probability curve.
                grid = np.arange(0, 12.01, .25).reshape(-1, 1)
                probs = model.predict_proba(grid)
                classes = list(model.classes_)
                pass_idx = classes.index(1) if 1 in classes else None
                curve = probs[:, pass_idx] if pass_idx is not None else np.zeros(len(grid))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=grid.flatten(), y=curve, mode="lines", name="Model estimate",
                                         line=dict(color="#3157d5", width=3), fill="tozeroy",
                                         fillcolor="rgba(49,87,213,.10)"))
                fig.add_vline(x=study_hours, line_dash="dash", line_color="#0f766e",
                              annotation_text=f"{clean_name}: {study_hours:g} hrs")
                fig.add_hline(y=.5, line_dash="dot", line_color="#98a2b3",
                              annotation_text="50% reference")
                fig.update_layout(height=310, margin=dict(l=8, r=8, t=30, b=8),
                                  paper_bgcolor="white", plot_bgcolor="white",
                                  xaxis_title="Study hours", yaxis_title="Estimated pass probability",
                                  yaxis=dict(range=[0, 1], tickformat=".0%"),
                                  font=dict(family="DM Sans", color="#344054"),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.session_state["last_single_prediction"] = {
                    "Student": clean_name,
                    "Study Hours": study_hours,
                    "Predicted Outcome": outcome,
                    "Estimated Pass Probability": probability,
                    "Estimated Pass Probability (%)": round(probability * 100, 1),
                }
                st.info("This is a model estimate for classroom demonstration—not a guarantee or an official student grade.")
            else:
                st.markdown('<div class="panel"><div class="panel-title">Prediction preview</div><p class="muted">Your student\'s result, estimated probability, and chart will appear here after you select <b>Predict outcome</b>.</p></div>', unsafe_allow_html=True)
            if "last_single_prediction" in st.session_state:
                last = st.session_state["last_single_prediction"]
                st.button(
                    f"➕ Save {last['Student']} to class predictions",
                    key="save_single",
                    on_click=save_last_prediction,
                    use_container_width=True,
                )
                st.markdown("")
                st.markdown("#### What you'll see")
                st.write("• Predicted Pass or Fail outcome")
                st.write("• Estimated probability of passing")
                st.write("• A chart showing where the student's study hours fall on the model curve")

    with tab_class:
        st.markdown("### Build a class list")
        st.write("Add several students at once, or upload a CSV file with `Student` and `Study Hours` columns.")
        with st.form("class_entry_form"):
            c1, c2 = st.columns([1.5, 1])
            with c1:
                names_text = st.text_area("Student names (one per line)", placeholder="Alex Santos\nJamie Cruz\nTaylor Reyes", height=115)
            with c2:
                hours_text = st.text_area("Study hours (one per line, same order)", placeholder="4\n6.5\n2", height=115)
            add_students = st.form_submit_button("Predict class list", type="primary", use_container_width=True)

        if add_students:
            names = [n.strip() for n in names_text.splitlines() if n.strip()]
            raw_hours = [h.strip() for h in hours_text.splitlines() if h.strip()]
            if not names or len(names) != len(raw_hours):
                st.error("Enter the same number of student names and study-hour values.")
            else:
                try:
                    hours_list = [float(h) for h in raw_hours]
                    if any(h < 0 or h > 12 for h in hours_list):
                        st.error("Each study-hour value must be between 0 and 12.")
                    else:
                        batch = predict_students(pd.DataFrame({"Student": names, "Study Hours": hours_list}))
                        st.session_state["class_predictions"] = pd.concat(
                            [st.session_state["class_predictions"], batch], ignore_index=True
                        )
                        st.success(f"Predictions created for {len(batch)} students.")
                except ValueError:
                    st.error("Study hours must be valid numbers, such as 3 or 4.5.")

        uploaded = st.file_uploader("Or upload a CSV", type=["csv"], help="Required columns: Student, Study Hours")
        if uploaded is not None:
            try:
                incoming = pd.read_csv(uploaded)
                incoming.columns = [c.strip() for c in incoming.columns]
                if not {"Student", "Study Hours"}.issubset(incoming.columns):
                    st.error("CSV must contain columns named Student and Study Hours.")
                elif incoming.empty:
                    st.warning("The CSV contains no student rows.")
                else:
                    incoming["Study Hours"] = pd.to_numeric(incoming["Study Hours"], errors="coerce")
                    if incoming["Study Hours"].isna().any() or ((incoming["Study Hours"] < 0) | (incoming["Study Hours"] > 12)).any():
                        st.error("All study-hour values must be numbers between 0 and 12.")
                    elif st.button("Predict uploaded students", type="primary"):
                        batch = predict_students(incoming[["Student", "Study Hours"]])
                        st.session_state["class_predictions"] = pd.concat(
                            [st.session_state["class_predictions"], batch], ignore_index=True
                        )
                        st.success(f"Predictions created for {len(batch)} uploaded students.")
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")

    st.divider()
    st.caption("Important: this demonstration model uses study hours only. It does not assess intelligence, effort, personal circumstances, or actual academic performance.")

elif page == "Class predictions":
    st.markdown("## Class prediction overview")
    st.write("Review, compare, and export predictions you've created during this session.")
    roster = st.session_state["class_predictions"].copy()
    if roster.empty:
        st.info("No class predictions yet. Go to **Dashboard → Add students to class** or predict an individual student and save them.")
    else:
        total = len(roster)
        pass_count = int((roster["Predicted Outcome"] == "Pass").sum())
        fail_count = total - pass_count
        avg_prob = float(roster["Estimated Pass Probability"].mean())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Students predicted", total)
        m2.metric("Predicted Pass", pass_count)
        m3.metric("Predicted Fail", fail_count)
        m4.metric("Average estimated pass probability", f"{avg_prob:.1%}")

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.markdown("#### Predicted outcomes")
            counts = roster["Predicted Outcome"].value_counts().reindex(["Pass", "Fail"], fill_value=0).rename_axis("Outcome").reset_index(name="Students")
            pie = px.pie(counts, names="Outcome", values="Students", hole=.62,
                         color="Outcome", color_discrete_map={"Pass": "#10b981", "Fail": "#f43f5e"})
            pie.update_layout(height=310, margin=dict(l=5,r=5,t=15,b=5), legend_title_text="")
            st.plotly_chart(pie, use_container_width=True)
        with chart_right:
            st.markdown("#### Study hours by student")
            bar = px.bar(roster, x="Student", y="Study Hours", color="Predicted Outcome",
                         color_discrete_map={"Pass": "#10b981", "Fail": "#f43f5e"},
                         hover_data={"Estimated Pass Probability (%)": True})
            bar.update_layout(height=310, margin=dict(l=5,r=5,t=15,b=60),
                              xaxis_title="", yaxis_title="Study hours", legend_title_text="")
            bar.update_xaxes(tickangle=-25)
            st.plotly_chart(bar, use_container_width=True)

        st.markdown("#### Student-by-student results")
        display = roster[["Student", "Study Hours", "Predicted Outcome", "Estimated Pass Probability (%)"]]
        st.dataframe(display, use_container_width=True, hide_index=True)
        csv_bytes = display.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download class results (CSV)", data=csv_bytes,
                           file_name="student_predictions.csv", mime="text/csv")
        if st.button("Clear class predictions", type="secondary"):
            st.session_state["class_predictions"] = pd.DataFrame(
                columns=["Student", "Study Hours", "Predicted Outcome",
                         "Estimated Pass Probability", "Estimated Pass Probability (%)"]
            )
            st.rerun()

elif page == "About the model":
    st.markdown("## About this prediction model")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel"><div class="panel-title">How the system works</div><br>1. A saved machine-learning model is loaded from a Joblib file.<br>2. The user enters a student name and study hours.<br>3. The model predicts Pass or Fail.<br>4. The dashboard displays the prediction and an estimated pass probability.<br>5. Multiple predictions can be compared and exported as CSV.</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">Responsible use</div><br>• The included training data are illustrative sample data.<br>• Study hours are the only feature used by this model.<br>• Attendance is not used for prediction.<br>• Probabilities are estimates, not promises.<br>• Do not use this demo to make real grading or high-impact decisions.</div>', unsafe_allow_html=True)
    st.warning("For a reliable real-world system, train and evaluate the model on an appropriate, representative dataset and report its performance before using predictions.")
