"""Exercise 3 (Termux-friendly) - web interface built with Flask, no Streamlit needed.
Run:  python web_app.py   then open http://127.0.0.1:8501 in your phone browser."""
import html
import joblib
import numpy as np
from flask import Flask, request
from model_utils import LogisticModel  # needed so joblib can load the model

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Pass Predictor</title>
<style>
body{{font-family:sans-serif;max-width:560px;margin:20px auto;padding:0 14px}}
input,button{{width:100%;padding:10px;margin:6px 0 12px;font-size:16px;box-sizing:border-box}}
button{{background:#2563eb;color:#fff;border:0;border-radius:6px}}
.ok{{background:#dcfce7;padding:12px;border-radius:6px}}
.bad{{background:#fee2e2;padding:12px;border-radius:6px}}
.warn{{background:#fef9c3;padding:12px;border-radius:6px}}
</style></head><body>
<h1>🎓 Student Pass Predictor</h1>
<form method="post">
<label>Student name</label><input name="name" value="{name}">
<label>Study hours per day</label><input name="hours" type="number" step="0.5" min="0" max="24" value="{hours}">
<button type="submit">Predict</button>
</form>{result}</body></html>"""


def chart_svg(hours, pred):
    """Pass-probability curve vs study hours, with the student's point marked."""
    W, H, P = 480, 260, 40
    xs = np.linspace(0, 12, 60)
    ps = model.predict_proba(xs.reshape(-1, 1))[:, 1]
    sx = lambda x: P + (min(x, 12) / 12) * (W - 2 * P)
    sy = lambda p: H - P - p * (H - 2 * P)
    pts = " ".join(f"{sx(x):.1f},{sy(p):.1f}" for x, p in zip(xs, ps))
    p_student = float(model.predict_proba([[hours]])[0][1])
    color = "#16a34a" if pred else "#dc2626"
    return f"""<svg viewBox="0 0 {W} {H}" width="100%">
<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#888"/>
<line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#888"/>
<line x1="{P}" y1="{sy(0.5)}" x2="{W-P}" y2="{sy(0.5)}" stroke="#aaa" stroke-dasharray="4"/>
<text x="{W-P}" y="{sy(0.5)-4}" font-size="11" text-anchor="end" fill="#666">50% = Pass line</text>
<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
<circle cx="{sx(hours)}" cy="{sy(p_student)}" r="7" fill="{color}"/>
<text x="{W/2}" y="{H-8}" font-size="12" text-anchor="middle">Study hours (0-12)</text>
<text x="12" y="{H/2}" font-size="12" transform="rotate(-90 12 {H/2})" text-anchor="middle">Pass probability</text>
</svg>"""


@app.route("/", methods=["GET", "POST"])
def index():
    name, hours, result = "", "3", ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        hours = request.form.get("hours", "")
        try:
            h = float(hours)
            if not 0 <= h <= 24:
                raise ValueError
        except ValueError:
            result = '<p class="warn">Please enter study hours between 0 and 24.</p>'
        else:
            if not name:
                result = '<p class="warn">Please enter the student\'s name.</p>'
            else:
                pred = int(model.predict([[h]])[0])
                prob = float(model.predict_proba([[h]])[0][1])
                n = html.escape(name)
                msg = (f'<p class="ok">{n} is predicted to <b>PASS</b> (probability {prob:.0%}).</p>' if pred
                       else f'<p class="bad">{n} is predicted to <b>FAIL</b> (pass probability {prob:.0%}).</p>')
                result = (msg + "<h3>How was this generated?</h3>"
                          f"<p>The app loaded a saved logistic regression model "
                          f"(<code>student_pass_model.joblib</code>) trained earlier on study hours. "
                          f"It used <b>{h:g} hours</b> to compute a pass probability of <b>{prob:.0%}</b>, "
                          f"and predicts Pass when that is 50% or more. The model was not retrained.</p>"
                          "<h3>Study hours vs predicted class</h3>" + chart_svg(h, pred))
    return PAGE.format(name=html.escape(name), hours=html.escape(hours), result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
