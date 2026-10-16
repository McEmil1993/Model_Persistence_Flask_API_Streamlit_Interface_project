from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / 'student_pass_model.joblib'


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')
    return joblib.load(MODEL_PATH)


@app.post('/predict')
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Request body must be a JSON object.'), 400

    missing = [name for name in ('study_hours', 'attendance_percentage') if name not in data]
    if missing:
        return jsonify(error=f'Missing required field(s): {", ".join(missing)}.'), 400

    try:
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance_percentage'])
    except (TypeError, ValueError):
        return jsonify(error='study_hours and attendance_percentage must be numeric.'), 400

    if not (study_hours == study_hours and abs(study_hours) != float('inf')):
        return jsonify(error='study_hours must be a finite number.'), 400
    if not (attendance == attendance and abs(attendance) != float('inf')):
        return jsonify(error='attendance_percentage must be a finite number.'), 400
    if not 0 <= attendance <= 100:
        return jsonify(error='attendance_percentage must be between 0 and 100.'), 400

    try:
        model = load_model()
        class_value = model.predict(pd.DataFrame({'study_hours': [study_hours]}))[0]
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 500
    except Exception:
        app.logger.exception('Model prediction failed')
        return jsonify(error='Prediction failed.'), 500

    label = 'Pass' if str(class_value) == '1' else 'Fail'
    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction=label,
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

