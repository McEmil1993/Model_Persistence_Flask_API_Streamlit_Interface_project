"""Exercise 3 - Streamlit interface, styled to match the Flask web app."""
import base64
import io
import re
from html import escape
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
FEATURES = ["study_hours", "attendance_percentage"]
GREEN, RED, BLUE, NAVY = "#138765", "#c7475d", "#4776e6", "#111d38"
CIRC = 282.74
TEMPLATE = ("name,study_hours,attendance_percentage\nAlex,5,85\nJordan,2,55\nMaria,3.5,70\nBen,1,45\n"
            "Liza,6,60\nCarlo,9,95\nAna,4,80\nNoel,2.5,90\n")
ALIASES = {"student": "name", "student_name": "name", "hours": "study_hours", "study_hours_per_day": "study_hours",
           "attendance": "attendance_percentage", "attendance_percent": "attendance_percentage"}

st.set_page_config(page_title="Student Pass Predictor | Academic Insights", page_icon="🎓", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "student_pass_model_v2.joblib")


model = load_model()
if "history" not in st.session_state:
    st.session_state.history = []

st.html("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Manrope:wght@500;700;800&display=swap');
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
.stApp {background:#f5f7fb; font-family:'DM Sans',sans-serif; overflow-x:hidden;}
.stApp label p, .stApp .stMarkdown p {font-family:'DM Sans',sans-serif;}
.block-container {max-width:1180px; padding:0 28px 24px !important;}
.topbar {background:#111d38; color:#fff; margin:0 calc(50% - 50vw) 30px; padding:18px calc(50vw - 50%);}
.topbar-in {display:flex; justify-content:space-between; align-items:center; max-width:1124px; margin:auto;}
.brand {display:flex; align-items:center; gap:12px; font:800 16px 'Manrope',sans-serif;}
.brand-mark {display:grid; place-items:center; width:42px; height:42px; border-radius:13px;
  background:linear-gradient(135deg,#ffd47a,#e9a83f); color:#17233e; font-size:21px;}
.brand small {display:block; font:500 11px 'DM Sans',sans-serif; letter-spacing:1.4px; text-transform:uppercase; color:#aab8d4;}
.status {display:inline-flex; align-items:center; gap:8px; border:1px solid #354363; background:#1a2949;
  border-radius:99px; padding:8px 12px; color:#d9e4f8; font-size:12px;}
.status i {width:7px; height:7px; border-radius:50%; background:#52d6a2; box-shadow:0 0 0 4px #52d6a21c;}
.eyebrow {color:#4776e6; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:1.8px; margin:0 0 7px;}
.h1 {font:800 clamp(27px,3vw,36px)/1.15 'Manrope',sans-serif; letter-spacing:-1.2px; color:#111d38; margin:0 0 9px;}
.sub {color:#748097; font-size:15px; max-width:640px; margin:0 0 24px;}
.st-key-form_card, .st-key-result_card, .st-key-batch_card {background:#fff; border:1px solid #e4e9f2; border-radius:20px;
  box-shadow:0 18px 55px rgba(24,42,77,.08); padding:27px 28px;}
.st-key-batch_card {margin-top:22px;}
[data-testid="stHorizontalBlock"] {align-items:stretch !important;}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {height:100%;}
[data-testid="stColumn"] [data-testid="stLayoutWrapper"]:has(> .st-key-form_card), [data-testid="stColumn"] [data-testid="stLayoutWrapper"]:has(> .st-key-result_card) {flex:1;}
.st-key-form_card, .st-key-result_card {height:100%;}
.card-head {display:flex; align-items:center; gap:12px; margin-bottom:18px;}
.icon {display:grid; place-items:center; width:42px; height:42px; border-radius:13px; background:#edf3ff; color:#4776e6; font-size:20px;}
.icon.green {background:#e9f7f1; color:#138765; margin-left:auto;}
.card-head b {font:800 17px 'Manrope',sans-serif; color:#1c2940; display:block;}
.card-head span.t {font-size:12px; color:#748097;}
.badge {margin-left:auto; font-size:10px; font-weight:800; letter-spacing:.8px; text-transform:uppercase;
  border-radius:7px; background:#f0f3f9; color:#77839a; padding:7px 9px;}
[data-testid="stImage"] {width:100% !important;} [data-testid="stImageContainer"] {width:100% !important; display:flex; justify-content:center;}
.verdict {text-align:center; font:800 23px 'Manrope',sans-serif; letter-spacing:-.5px; margin:2px 0 6px; color:#111d38;}
.verdict.pass {color:#138765;} .verdict.fail {color:#c7475d;}
.explain {text-align:center; color:#748097; font-size:13px; max-width:460px; margin:0 auto 18px;}
.stats {display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:18px;}
.stat {border:1px solid #e4e9f2; background:#fbfcfe; border-radius:12px; padding:12px 14px;}
.stat .l {font-size:10px; text-transform:uppercase; letter-spacing:1px; font-weight:800; color:#8a95a8; margin-bottom:4px;}
.stat .v {font:800 18px 'Manrope',sans-serif; color:#111d38;} .stat .v small {font:600 11px 'DM Sans',sans-serif; color:#748097;}
.chart-title {font:800 14px 'Manrope',sans-serif; color:#1c2940; margin:6px 0 2px;}
.chart-sub {font-size:12px; color:#748097; margin:0 0 4px;}
.hist {border-top:1px solid #e4e9f2; margin-top:14px; padding-top:14px;}
.hist h4 {font:800 12px 'Manrope',sans-serif; text-transform:uppercase; letter-spacing:1px; margin:0 0 6px; color:#1c2940;}
.hist .row {display:flex; justify-content:space-between; align-items:center; padding:9px 0; border-top:1px solid #f0f2f7; font-size:12px; color:#1c2940;}
.hist .meta {display:block; color:#748097; font-size:11px;} .hist .empty {color:#748097; font-size:12px; margin:0;}
.chip {display:inline-block; font-size:10px; font-weight:800; padding:5px 9px; border-radius:99px; white-space:nowrap;}
.chip.pass {background:#e7f7f0; color:#138765;} .chip.fail {background:#fff0f2; color:#c7475d;} .chip.neutral {background:#f0f3f9; color:#64718a;}
.hint {font-size:11px; color:#929db0; margin:-8px 0 14px;}
.st-key-form_card [data-testid="stFormSubmitButton"] button {background:#4776e6; border:0; color:#fff; border-radius:12px;
  padding:.7rem 1rem; font-weight:800; box-shadow:0 8px 18px #4776e62b;}
.st-key-form_card [data-testid="stFormSubmitButton"] button:hover {background:#3867d4;}
.st-key-batch_card [data-testid="stDownloadButton"] button {border:1px solid #dce3ef; border-radius:10px; background:#fff; color:#34425b; font-weight:700;}
.st-key-batch_card [data-testid="stFileUploaderDropzone"] {border:2px dashed #cbd5e4; background:#fafbfe; border-radius:12px;}
.sumrow {display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 12px;}
.tablewrap {overflow:auto; max-height:560px; border:1px solid #e4e9f2; border-radius:12px;}
.tablewrap table {width:100%; border-collapse:collapse; font-size:12px; color:#1c2940;}
.tablewrap th, .tablewrap td {padding:12px 14px; text-align:left; white-space:nowrap;}
.tablewrap th {position:sticky; top:0; background:#f7f9fc; color:#65718a; font-size:10px; text-transform:uppercase; letter-spacing:.7px;}
.tablewrap tr + tr td {border-top:1px solid #edf0f6;}
.hist-head {display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;}
.hist-head h4 {margin:0 !important;}
.count {font-size:10px; font-weight:700; color:#748097; background:#f0f3f9; border-radius:99px; padding:4px 9px;}
.rcard {display:flex; align-items:center; gap:12px; border:1px solid #e4e9f2; border-left-width:4px; border-radius:14px;
  padding:12px 14px; margin-bottom:10px; background:#fbfcfe; box-shadow:0 2px 8px rgba(24,42,77,.04);}
.rcard.pass {border-left-color:#138765;} .rcard.fail {border-left-color:#c7475d;}
.av {display:grid; place-items:center; flex:0 0 38px; width:38px; height:38px; border-radius:50%; font:800 13px 'Manrope',sans-serif;}
.av.pass {background:#e7f7f0; color:#138765;} .av.fail {background:#fff0f2; color:#c7475d;}
.rinfo {flex:1; min-width:0;}
.rinfo b {display:block; font:700 14px 'DM Sans',sans-serif; color:#1c2940; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.rinfo span {display:block; font-size:11px; color:#748097;}
.mbar {height:5px; background:#e9edf5; border-radius:99px; margin-top:7px; overflow:hidden;}
.mbar i {display:block; height:100%; border-radius:99px;}
.rcard.pass .mbar i {background:#138765;} .rcard.fail .mbar i {background:#c7475d;}
.rres {text-align:right;} .rres strong {display:block; font:800 16px 'Manrope',sans-serif; color:#111d38; margin-top:5px;}
.rempty {border:1.5px dashed #cbd5e4; border-radius:14px; padding:18px; text-align:center; color:#748097; font-size:12px; background:#fafbfe;}
.note {text-align:center; color:#8994a7; font-size:11px; margin-top:22px;}
@media (max-width:800px) {.block-container {padding:0 14px 20px !important;} .st-key-form_card, .st-key-result_card, .st-key-batch_card {padding:22px 18px;}}
</style>""")

st.html("""<div class="topbar"><div class="topbar-in">
  <div class="brand"><span class="brand-mark">✦</span><span>Student Pass Predictor<small>Academic insights</small></span></div>
  <div class="status"><i></i>Model loaded</div></div></div>
<p class="eyebrow">Student analytics / Overview</p>
<p class="h1">Understand the likelihood of success.</p>
<p class="sub">Enter a student's study habits and attendance to generate a model-based estimate, or score a whole class from a CSV file.</p>""")


def gauge(prob=None, passed=None):
    """Half-circle gauge as SVG text (st.image renders it safely)."""
    offset = CIRC if prob is None else CIRC * (1 - prob)
    color = "#cbd3e1" if prob is None else (GREEN if passed else RED)
    num = "--" if prob is None else f"{prob:.0%}"
    f = "font-family=\"Manrope,'Segoe UI',Arial,sans-serif\""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 135">
      <path d="M 20 115 A 90 90 0 0 1 200 115" fill="none" stroke="#edf0f6" stroke-width="16" stroke-linecap="round"/>
      <path d="M 20 115 A 90 90 0 0 1 200 115" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"
            stroke-dasharray="{CIRC}" stroke-dashoffset="{offset}"/>
      <line x1="110" y1="15" x2="110" y2="35" stroke="{NAVY}" stroke-width="2"/>
      <text x="110" y="9" text-anchor="middle" {f} font-size="9" font-weight="700" fill="#8a95a8">PASS LINE 50%</text>
      <text x="20" y="130" text-anchor="middle" {f} font-size="9" font-weight="700" fill="#8a95a8">0%</text>
      <text x="200" y="130" text-anchor="middle" {f} font-size="9" font-weight="700" fill="#8a95a8">100%</text>
      <text x="110" y="92" text-anchor="middle" {f} font-size="40" font-weight="800" fill="{NAVY}">{num}</text>
      <text x="110" y="112" text-anchor="middle" {f} font-size="10" font-weight="600" fill="#748097">ESTIMATED PASS CHANCE</text></svg>"""


def gauge_img(prob=None, passed=None):
    """The gauge as a centred <img> (SVG as a data URI)."""
    b64 = base64.b64encode(gauge(prob, passed).encode()).decode()
    return ('<div style="text-align:center"><img alt="Pass probability gauge" '
            'style="display:block;margin:0 auto;width:340px;max-width:100%;height:auto" '
            f'src="data:image/svg+xml;base64,{b64}"></div>')


def stats_html(hours="—", attendance="—"):
    return (f'<div class="stats"><div class="stat"><div class="l">Study time</div><div class="v">{hours} <small>hours/day</small></div></div>'
            f'<div class="stat"><div class="l">Attendance</div><div class="v">{attendance} <small>percent</small></div></div></div>')


def make_chart(name, hours, attendance, prob, passed):
    max_h = max(12, int(np.ceil(hours)) + 1)
    curve = pd.DataFrame({FEATURES[0]: np.linspace(0, max_h, 121)})
    curve[FEATURES[1]] = attendance
    curve["pass_probability"] = model.predict_proba(curve[FEATURES])[:, 1]
    student = pd.DataFrame({"study_hours": [hours], "pass_probability": [prob], "student": [name]})
    x = alt.X("study_hours:Q", title="Study hours per day", scale=alt.Scale(domain=[0, max_h]))
    y = alt.Y("pass_probability:Q", title="Chance of passing", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]))
    zones = pd.DataFrame({"lo": [0, 0.5], "hi": [0.5, 1], "zone": ["At risk", "Pass"]})
    bands = alt.Chart(zones).mark_rect(opacity=0.10).encode(
        y="lo:Q", y2="hi:Q", color=alt.Color("zone:N", legend=None, scale=alt.Scale(domain=["At risk", "Pass"], range=[RED, GREEN])))
    line = alt.Chart(curve).mark_line(strokeWidth=3, color=BLUE).encode(x=x, y=y)
    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[5, 4], color=NAVY).encode(y="y:Q")
    dot = alt.Chart(student).mark_point(size=260, filled=True, opacity=1, stroke="white", strokeWidth=2,
                                        color=GREEN if passed else RED).encode(
        x=x, y=y, tooltip=["student", "study_hours", alt.Tooltip("pass_probability:Q", format=".0%")])
    label = alt.Chart(student).mark_text(dy=-18, fontWeight="bold", color=NAVY).encode(x=x, y=y, text="student:N")
    return (alt.layer(bands, line, rule, dot, label).properties(height=300)
            .configure_axis(labelColor="#748097", titleColor="#748097", gridColor="#e4e9f2", domain=False)
            .configure_axisX(grid=False)
            .configure_view(strokeWidth=0))


def score_csv(raw: bytes):
    """Score an uploaded class list. Handles Excel encodings, ';' separators and friendly headers."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("The file could not be read.")
    header = text.splitlines()[0] if text.strip() else ""
    df = pd.read_csv(io.StringIO(text), sep=";" if header.count(";") > header.count(",") else ",")
    norm = lambda c: ALIASES.get(re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"),
                                 re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"))
    df.columns = [norm(c) for c in df.columns]
    missing = {"name", *FEATURES} - set(df.columns)
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(sorted(missing)) + ". Found: " + ", ".join(map(str, df.columns)))
    if not 1 <= len(df) <= 1000:
        raise ValueError("The file must have between 1 and 1,000 students.")
    num = lambda v: pd.to_numeric(str(v).replace("%", "").strip(), errors="coerce")
    rows = []
    for _, r in df.iterrows():
        h, a = num(r[FEATURES[0]]), num(r[FEATURES[1]])
        name = "" if pd.isna(r["name"]) else str(r["name"]).strip()
        if pd.isna(h) or pd.isna(a) or not 0 <= h <= 24 or not 0 <= a <= 100:
            rows.append(dict(name=name, hours=None if pd.isna(h) else float(h), att=None if pd.isna(a) else float(a), result="invalid", prob=None))
            continue
        s = pd.DataFrame({FEATURES[0]: [float(h)], FEATURES[1]: [float(a)]})
        rows.append(dict(name=name, hours=float(h), att=float(a), result="pass" if int(model.predict(s)[0]) == 1 else "fail",
                         prob=float(model.predict_proba(s)[0][1])))
    return rows


left, right = st.columns([0.92, 1.25], gap="medium")

with left:
    with st.container(key="form_card"):
        st.html('<div class="card-head"><span class="icon">＋</span><div><b>Student details</b>'
                '<span class="t">Enter the information below to run an estimate.</span></div></div>')
        with st.form("predict", border=False):
            name = st.text_input("Student name", placeholder="e.g. Alex")
            hours = st.slider("Study hours per day", 0.0, 12.0, 5.0, 0.5)
            st.html('<p class="hint">Average time spent studying each day.</p>')
            attendance = st.slider("Attendance percentage", 0, 100, 85)
            st.html('<p class="hint">Percentage of classes attended.</p>')
            submitted = st.form_submit_button("Generate prediction  →", type="primary", width="stretch")
        hist_slot = st.empty()

result = None
if submitted:
    if not name.strip():
        with left:
            st.warning("Enter the student's name to continue.")
    else:
        sample = pd.DataFrame({FEATURES[0]: [hours], FEATURES[1]: [attendance]})
        passed = int(model.predict(sample)[0]) == 1
        prob = float(model.predict_proba(sample)[0][1])
        result = dict(name=name.strip(), hours=hours, attendance=attendance, prob=prob, passed=passed)
        st.session_state["last"] = result
        st.session_state.history = ([result] + st.session_state.history)[:5]

result = st.session_state.get("last")

with right:
    with st.container(key="result_card"):
        badge = "Prediction complete" if result else "Awaiting input"
        st.html(f'<div class="card-head"><div><b>Prediction overview</b><span class="t">Your model result will appear here.</span>'
                f'</div><span class="badge">{badge}</span></div>')
        if result:
            who, p, ok = escape(result["name"]), result["prob"], result["passed"]
            st.html(gauge_img(p, ok))
            st.html(f'<p class="verdict {"pass" if ok else "fail"}">{who} is predicted to {"pass" if ok else "need support"}</p>'
                    f'<p class="explain">A logistic regression model, trained earlier on historical student data and loaded from a saved file, '
                    f'combined {result["hours"]:g} study hours and {result["attendance"]}% attendance into a {p:.0%} chance of passing. '
                    f'A probability of 50% or more is classified as PASS. The model is not retrained for each prediction.</p>'
                    + stats_html(f'{result["hours"]:g}', f'{result["attendance"]}%')
                    + f'<p class="chart-title">Study hours vs. predicted result</p>'
                    f'<p class="chart-sub">The blue line is the model\'s pass chance at {result["attendance"]}% attendance. The dot is {who}.</p>')
            st.altair_chart(make_chart(result["name"], result["hours"], result["attendance"], p, ok), width="stretch")
        else:
            st.html(gauge_img())
            st.html('<p class="verdict">Ready when you are</p><p class="explain">Add the student\'s details and select '
                    '"Generate prediction" to see the result and how it was calculated.</p>' + stats_html())

def initials(full_name):
    parts = full_name.split()
    return escape((parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper())


cards = "".join(
    f'<div class="rcard {"pass" if h["passed"] else "fail"}"><span class="av {"pass" if h["passed"] else "fail"}">{initials(h["name"])}</span>'
    f'<div class="rinfo"><b>{escape(h["name"])}</b><span>{h["hours"]:g} h/day · {h["attendance"]}% attendance</span>'
    f'<div class="mbar"><i style="width:{h["prob"] * 100:.0f}%"></i></div></div>'
    f'<div class="rres"><span class="chip {"pass" if h["passed"] else "fail"}">{"Pass" if h["passed"] else "At risk"}</span>'
    f'<strong>{h["prob"]:.0%}</strong></div></div>'
    for h in st.session_state.history
) or '<div class="rempty">No students yet. Each prediction you run appears here as a card.</div>'
hist_slot.html(f'<div class="hist"><div class="hist-head"><h4>Recent students</h4>'
               f'<span class="count">{len(st.session_state.history)} of 5</span></div>{cards}</div>')

# ---------- Bulk assessment ----------
with st.container(key="batch_card"):
    st.html('<div class="card-head"><div><p class="eyebrow" style="margin:0 0 4px">Bulk assessment</p><b style="font-size:18px">Score a whole class</b>'
            '<span class="t">Upload a CSV with <strong style="color:#1c2940">name</strong>, <strong style="color:#1c2940">study_hours</strong> '
            'and <strong style="color:#1c2940">attendance_percentage</strong> (up to 1,000 students).</span></div><span class="icon green">▤</span></div>')
    c1, c2 = st.columns([0.3, 0.7], vertical_alignment="center")
    c1.download_button("↓ Download CSV template", TEMPLATE, "class-template.csv", "text/csv")
    upload = c2.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
    if upload is not None:
        try:
            batch = score_csv(upload.getvalue())
        except Exception as exc:
            st.error(str(exc))
        else:
            n_pass = sum(r["result"] == "pass" for r in batch)
            n_fail = sum(r["result"] == "fail" for r in batch)
            n_bad = sum(r["result"] == "invalid" for r in batch)
            chips = (f'<span class="chip neutral">{len(batch)} students total</span><span class="chip pass">{n_pass} predicted to pass</span>'
                     f'<span class="chip fail">{n_fail} may need support</span>'
                     + (f'<span class="chip neutral">{n_bad} rows need checking</span>' if n_bad else ""))
            body = "".join(
                f'<tr><td>{escape(r["name"] or "(no name)")}</td><td>{"—" if r["hours"] is None else f"{r['hours']:g}"}</td>'
                f'<td>{"—" if r["att"] is None else f"{r['att']:g}%"}</td>'
                f'<td><span class="chip {"pass" if r["result"] == "pass" else "fail" if r["result"] == "fail" else "neutral"}">'
                f'{"Pass" if r["result"] == "pass" else "At risk" if r["result"] == "fail" else "Check data"}</span></td>'
                f'<td>{"—" if r["prob"] is None else f"{r['prob']:.0%}"}</td></tr>' for r in batch)
            st.html(f'<div class="sumrow">{chips}</div><div class="tablewrap"><table><thead><tr><th>Name</th><th>Study hours</th>'
                    f'<th>Attendance</th><th>Result</th><th>Pass chance</th></tr></thead><tbody>{body}</tbody></table></div>')
            safe = lambda v: '"' + ("'" + str(v) if re.match(r"^[=+\-@]", str(v)) else str(v)).replace('"', '""') + '"'
            out = "name,study_hours,attendance_percentage,result,pass_probability\n" + "\n".join(
                ",".join(safe(v) for v in (r["name"], r["hours"] if r["hours"] is not None else "", r["att"] if r["att"] is not None else "",
                                           r["result"], round(r["prob"], 3) if r["prob"] is not None else "")) for r in batch)
            st.download_button("↓ Download results (CSV)", out, "class-results.csv", "text/csv")

st.html('<p class="note"><b>Note:</b> Predictions are estimates generated by a logistic regression model trained on historical '
        'student data. They are for guidance only and should not be the sole basis for academic decisions.</p>')