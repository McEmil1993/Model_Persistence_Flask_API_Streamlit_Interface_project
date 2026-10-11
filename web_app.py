"""Exercise 3 (Termux-friendly version) - web interface built with Flask.

Inputs: student name, study hours, attendance percentage.
Uses the model trained on BOTH study hours and attendance.
Needs only numpy, joblib and flask.

Run:  python web_app.py
Then open  http://localhost:5001  in your phone browser.
"""
import joblib
import numpy as np
from flask import Flask, render_template_string, request

import model_lib  # noqa: F401  (lets joblib rebuild the model class)

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pass predictor</title>
<style>
  :root {
    --paper: #eef3f7;
    --grid: #d3dee8;
    --card: #ffffff;
    --ink: #14213d;
    --muted: #5b6b82;
    --pen: #2457d6;
    --red: #d7263d;
    --green: #1b7f5c;
    --line: #c5d2df;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #0f1726; --grid: #1a2538; --card: #16213a; --ink: #e8eefc;
      --muted: #9fb0cc; --pen: #7aa2ff; --red: #ff6b7d; --green: #4cc9a0; --line: #2c3b58;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; color: var(--ink); background-color: var(--paper);
    background-image:
      linear-gradient(var(--grid) 1px, transparent 1px),
      linear-gradient(90deg, var(--grid) 1px, transparent 1px);
    background-size: 24px 24px;
    font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  main { max-width: 600px; margin: 0 auto; padding: 28px 16px 48px; }
  h1 {
    font: 700 34px/1.1 Georgia, "Times New Roman", serif;
    margin: 0 0 6px; letter-spacing: -0.5px;
  }
  .lede { margin: 0 0 22px; color: var(--muted); max-width: 46ch; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 18px; margin-bottom: 18px;
  }
  label { display: block; font-weight: 600; margin-bottom: 4px; }
  .field { margin-bottom: 18px; }
  .row { display: flex; align-items: center; gap: 12px; }
  .row input[type=range] { flex: 1; accent-color: var(--pen); }
  input[type=text], input[type=number] {
    width: 100%; padding: 10px 12px; font-size: 16px; color: var(--ink);
    background: var(--paper); border: 1px solid var(--line); border-radius: 6px;
  }
  .row input[type=number] { width: 84px; }
  input:focus-visible, button:focus-visible {
    outline: 3px solid var(--pen); outline-offset: 2px;
  }
  .hint { color: var(--muted); font-size: 14px; margin-top: 2px; }
  button {
    width: 100%; padding: 13px; font-size: 17px; font-weight: 700; cursor: pointer;
    color: #fff; background: var(--pen); border: 0; border-radius: 8px;
  }
  button:hover { filter: brightness(1.08); }
  .error {
    border-left: 5px solid var(--red); background: var(--card); padding: 12px 14px;
    border-radius: 6px; margin-bottom: 18px;
  }
  .verdict { display: flex; align-items: center; gap: 18px; }
  .stamp {
    flex: none; transform: rotate(-6deg); text-align: center;
    border: 4px solid; border-radius: 10px; padding: 8px 16px;
    font: 800 34px/1 Georgia, serif; letter-spacing: 2px;
    animation: stamp .35s ease-out;
  }
  .stamp small { display: block; font: 600 13px/1.4 system-ui, sans-serif; letter-spacing: 0; }
  .stamp.pass { color: var(--green); border-color: var(--green); }
  .stamp.fail { color: var(--red); border-color: var(--red); }
  @keyframes stamp { from { transform: rotate(-6deg) scale(1.6); opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .stamp { animation: none; } }
  .verdict p { margin: 0; }
  h2 { font: 700 20px/1.2 Georgia, serif; margin: 0 0 10px; }
  .card p { margin: 0 0 10px; }
  .card p:last-child { margin-bottom: 0; }
  svg { width: 100%; height: auto; display: block; }
  svg text { fill: var(--muted); font: 14px system-ui, sans-serif; }
  svg .axis { stroke: var(--muted); }
  svg .cut { stroke: var(--muted); stroke-dasharray: 5 4; }
  svg .curve { fill: none; stroke: var(--pen); stroke-width: 3; }
  svg .dot { fill: var(--red); stroke: var(--card); stroke-width: 2; }
  .chart-title { font-weight: 600; margin: 14px 0 2px; }
  .chart-title:first-of-type { margin-top: 0; }
</style>
</head>
<body>
<main>
  <h1>Pass predictor</h1>
  <p class="lede">Enter a student's study time and attendance to see whether they are likely to pass.</p>

  <form method="post" class="card" novalidate>
    <div class="field">
      <label for="name">Student name</label>
      <input type="text" id="name" name="name" value="{{ name }}" autocomplete="off">
    </div>
    <div class="field">
      <label for="hours">Study hours per day</label>
      <div class="row">
        <input type="range" id="hours_r" min="0" max="12" step="0.5" value="{{ hours }}" aria-label="Study hours slider">
        <input type="number" id="hours" name="hours" min="0" max="24" step="0.5" value="{{ hours }}">
      </div>
    </div>
    <div class="field">
      <label for="attendance">Attendance percentage</label>
      <div class="row">
        <input type="range" id="attendance_r" min="0" max="100" step="1" value="{{ attendance }}" aria-label="Attendance slider">
        <input type="number" id="attendance" name="attendance" min="0" max="100" step="1" value="{{ attendance }}">
      </div>
      <div class="hint">Share of classes attended, from 0 to 100.</div>
    </div>
    <button type="submit">Predict result</button>
  </form>

  {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}

  {% if result %}
  <section class="card" aria-live="polite">
    <div class="verdict">
      <div class="stamp {{ 'pass' if result.passed else 'fail' }}">
        {{ 'PASS' if result.passed else 'FAIL' }}
        <small>pass probability {{ result.percent }}%</small>
      </div>
      <p><b>{{ name }}</b> studies {{ hours }} hours a day and attends {{ attendance }}% of classes.
      The model predicts they will <b>{{ 'pass' if result.passed else 'fail' }}</b>.</p>
    </div>
  </section>

  <section class="card">
    <h2>How this prediction was made</h2>
    <p>A logistic regression model learned from past student records: study
    hours, attendance, and whether the student passed. It combines the two
    numbers you entered into one pass probability.</p>
    <p>A probability of 50% or higher gives PASS, anything lower gives FAIL.
    The trained model was saved with joblib and loaded here, so it is not
    retrained for each prediction.</p>
  </section>

  <section class="card">
    <h2>Study hours and attendance vs the result</h2>
    <div class="chart-title">Pass probability by study hours (at {{ attendance }}% attendance)</div>
    {{ chart_hours|safe }}
    <div class="chart-title">Pass probability by attendance (at {{ hours }} study hours)</div>
    {{ chart_att|safe }}
    <p class="hint">Blue line: probability of passing. Dashed line: the 50% cut-off. Red dot: this student.</p>
  </section>
  {% endif %}
</main>
<script>
  // keep each slider and its number box in sync
  [["hours", "hours_r"], ["attendance", "attendance_r"]].forEach(function (pair) {
    var num = document.getElementById(pair[0]), rng = document.getElementById(pair[1]);
    num.addEventListener("input", function () { rng.value = num.value; });
    rng.addEventListener("input", function () { num.value = rng.value; });
  });
</script>
</body>
</html>
"""


def make_chart(xs, probs, cur_x, cur_p, x_max, ticks, x_label):
    """Draw an inline SVG line chart (no extra libraries needed)."""
    w, h, pl, pr, pt, pb = 500, 240, 44, 14, 12, 42
    sx = lambda x: pl + (x / x_max) * (w - pl - pr)
    sy = lambda p: h - pb - p * (h - pt - pb)

    line = " ".join(f"{sx(x):.1f},{sy(p):.1f}" for x, p in zip(xs, probs))
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{x_label} against pass probability">']
    out.append(f'<line class="axis" x1="{pl}" y1="{h-pb}" x2="{w-pr}" y2="{h-pb}"/>')
    out.append(f'<line class="axis" x1="{pl}" y1="{pt}" x2="{pl}" y2="{h-pb}"/>')
    out.append(f'<line class="cut" x1="{pl}" y1="{sy(0.5):.1f}" x2="{w-pr}" y2="{sy(0.5):.1f}"/>')
    out.append(f'<polyline class="curve" points="{line}"/>')
    out.append(f'<circle class="dot" cx="{sx(min(cur_x, x_max)):.1f}" cy="{sy(cur_p):.1f}" r="7"/>')
    for t in ticks:
        out.append(f'<text x="{sx(t):.1f}" y="{h-pb+16}" text-anchor="middle">{t}</text>')
    for p in (0, 0.5, 1):
        out.append(f'<text x="{pl-6}" y="{sy(p)+4:.1f}" text-anchor="end">{int(p*100)}%</text>')
    out.append(f'<text x="{(pl+w-pr)/2}" y="{h-8}" text-anchor="middle">{x_label}</text>')
    out.append("</svg>")
    return "".join(out)


def parse_number(text, low, high, label):
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.")
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g}.")
    return value


@app.route("/", methods=["GET", "POST"])
def index():
    name, hours, attendance = "", "4", "75"
    result, error, chart_hours, chart_att = None, None, "", ""

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        hours = request.form.get("hours", "").strip()
        attendance = request.form.get("attendance", "").strip()
        try:
            h = parse_number(hours, 0, 24, "Study hours")
            a = parse_number(attendance, 0, 100, "Attendance")
            if not name:
                raise ValueError("Enter the student's name.")
        except ValueError as exc:
            error = str(exc)
        else:
            prob = float(model.predict_proba(np.array([[h, a]]))[0])
            result = {"passed": prob >= 0.5, "percent": round(prob * 100)}

            xs = np.linspace(0, 12, 61)
            ps = model.predict_proba(np.column_stack([xs, np.full_like(xs, a)]))
            chart_hours = make_chart(xs, ps, h, prob, 12, range(0, 13, 2), "Study hours per day")

            ys = np.linspace(0, 100, 51)
            qs = model.predict_proba(np.column_stack([np.full_like(ys, h), ys]))
            chart_att = make_chart(ys, qs, a, prob, 100, range(0, 101, 20), "Attendance (%)")

            hours, attendance = f"{h:g}", f"{a:g}"

    return render_template_string(
        PAGE, name=name, hours=hours, attendance=attendance, result=result,
        error=error, chart_hours=chart_hours, chart_att=chart_att,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
