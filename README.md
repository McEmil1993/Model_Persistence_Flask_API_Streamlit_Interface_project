# Student Pass Predictor

Small system with 3 parts: model training, Flask API, Streamlit UI.

## Files

- `train_model.py` — trains a 2-feature Logistic Regression on [study hours, attendance] → pass/fail (40 labeled samples; pass needs decent hours AND attendance), saves `student_pass_model.joblib` via `joblib.dump()`, reloads it via `joblib.load()`, predicts sample combos with no retraining.
- `app.py` — Flask API on port 5000. `POST /predict` accepts `study_hours`, `attendance_percentage` (validated 0–100), optional `student_name`. Returns `study_hours`, `attendance_percentage`, `student_name`, `prediction` (0/1), `prediction_label` (`Pass`/`Fail`), `result` (`PASS`/`FAIL`), `confidence` % (from `predict_proba`). Both inputs shape the prediction.
- `streamlit_app.py` — Streamlit UI that calls the Flask API: dark-mode toggle (dark default), student name + study-hours + attendance inputs, Predict button, result card, metric cards (prediction / confidence / hours), how-it-was-generated box, Altair decision-curve chart (line + your diamond + 5.5h boundary, themed per mode).

## Run (in order — the UI needs the API)

```bash
python train_model.py
python app.py
# test: curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"student_name\":\"Alex\",\"study_hours\":8,\"attendance_percentage\":85}"

python -m streamlit run streamlit_app.py
```
