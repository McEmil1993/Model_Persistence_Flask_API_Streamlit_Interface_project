"""Student Pass Predictor - web page, single prediction API and batch (CSV) scoring."""
import io
import logging
import re
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent
FEATURES = ["study_hours", "attendance_percentage"]
MAX_ROWS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predictor")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # reject uploads over 1 MB
model = joblib.load(BASE_DIR / "models" / "student_pass_model_v2.joblib")  # loaded once


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'"
    )
    return resp


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found"), 404


@app.errorhandler(405)
def bad_method(_):
    return jsonify(error="Method not allowed"), 405


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File is too large (limit 1 MB)"), 413


@app.errorhandler(Exception)
def unexpected(exc):
    log.exception("Unexpected error")
    return jsonify(error="Something went wrong on the server"), 500


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


def _score(hours, attendance):
    sample = pd.DataFrame({FEATURES[0]: [hours], FEATURES[1]: [attendance]})
    return int(model.predict(sample)[0]), float(model.predict_proba(sample)[0][1])


def _curve(attendance):
    """Model pass chance for 0-24 study hours at a given attendance (draws the chart line)."""
    grid = [h / 2 for h in range(0, 49)]
    probs = model.predict_proba(pd.DataFrame({FEATURES[0]: grid, FEATURES[1]: attendance}))[:, 1]
    return [{"study_hours": h, "pass_probability": round(float(p), 3)} for h, p in zip(grid, probs)]


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON body"), 400
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify(error="study_hours and attendance_percentage are required"), 400
    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify(error="Both values must be numbers"), 400
    if not 0 <= hours <= 24:
        return jsonify(error="study_hours must be between 0 and 24"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    pred, prob = _score(hours, attendance)
    log.info("single prediction served")
    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction="pass" if pred == 1 else "fail",
        pass_probability=round(prob, 3),
        curve=_curve(attendance),
    )


ALIASES = {
    "student": "name", "student_name": "name", "full_name": "name",
    "hours": "study_hours", "study_hours_per_day": "study_hours", "study_time": "study_hours",
    "attendance": "attendance_percentage", "attendance_percent": "attendance_percentage",
    "attendance_rate": "attendance_percentage",
}


def _norm(col):
    """'Study Hours', ' study-hours ' and a hidden BOM all become 'study_hours'."""
    name = re.sub(r"[^a-z0-9]+", "_", str(col).replace("\ufeff", "").strip().lower()).strip("_")
    return ALIASES.get(name, name)


def _read_csv(file):
    """Read CSVs saved by Excel: handles BOM, Windows encodings and ; separators."""
    raw = file.read()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("unreadable encoding")
    header = text.splitlines()[0] if text.strip() else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def _num(value):
    return pd.to_numeric(str(value).replace("%", "").strip(), errors="coerce")


@app.post("/predict-batch")
def predict_batch():
    file = request.files.get("file")
    if file is None:
        return jsonify(error="Upload a CSV file in the 'file' field"), 400
    try:
        df = _read_csv(file)
    except Exception:
        return jsonify(error="That file could not be read as a CSV"), 400

    df.columns = [_norm(c) for c in df.columns]
    missing = {"name", *FEATURES} - set(df.columns)
    if missing:
        return jsonify(error="Missing column(s): " + ", ".join(sorted(missing))
                       + ". Found: " + ", ".join(map(str, df.columns))), 400
    if not 1 <= len(df) <= MAX_ROWS:
        return jsonify(error=f"The file must have between 1 and {MAX_ROWS} students"), 400

    rows, counts = [], {"pass": 0, "fail": 0, "invalid": 0}
    for _, r in df.iterrows():
        name = "" if pd.isna(r["name"]) else str(r["name"]).strip()
        hours = _num(r[FEATURES[0]])
        att = _num(r[FEATURES[1]])
        if pd.isna(hours) or pd.isna(att) or not 0 <= hours <= 24 or not 0 <= att <= 100:
            counts["invalid"] += 1
            rows.append({"name": name, "study_hours": None if pd.isna(hours) else float(hours),
                         "attendance_percentage": None if pd.isna(att) else float(att),
                         "prediction": "invalid", "pass_probability": None})
            continue
        pred, prob = _score(float(hours), float(att))
        label = "pass" if pred == 1 else "fail"
        counts[label] += 1
        rows.append({"name": name, "study_hours": float(hours), "attendance_percentage": float(att),
                     "prediction": label, "pass_probability": round(prob, 3)})

    log.info("batch scored: %s", counts)
    curve, curve_att = [], None
    valid = [r for r in rows if r["prediction"] != "invalid"]
    if valid:  # the model's pass chance across study hours at the class-average attendance
        curve_att = round(sum(r["attendance_percentage"] for r in valid) / len(valid), 1)
        curve = _curve(curve_att)

    return jsonify(summary=counts, total=len(rows), results=rows,
                   curve=curve, curve_attendance=curve_att)


if __name__ == "__main__":
    app.run(debug=True)