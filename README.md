# Student Pass Exercises

2-feature `LogisticRegression` pipeline (`study_hours`, `attendance_percentage` -> pass/fail) covering model persistence, Flask API, and Streamlit UI.

## Setup

```powershell
cd C:\Users\ACER\James_Exercise
pip install scikit-learn joblib flask streamlit pandas
```

Files: `train_model.py`, `app.py`, `streamlit_app.py`, `student_pass_model.joblib`.

## Exercise 1 — Model Persistence (`train_model.py`)

- Builds synthetic data (seed 42, 300 rows): `score = 0.8*hours + 0.05*attendance + noise`, label = `score > 5.0`.
- Trains `StandardScaler + LogisticRegression` pipeline with `train_test_split` (70/30).
- `joblib.dump()` saves to `student_pass_model.joblib`; `joblib.load()` reloads it.
- Makes 3 predictions with different study hours: `(2, 60)` -> FAIL, `(5, 75)` -> PASS, `(8, 90)` -> PASS.
- No retraining needed because dump persists scaler stats + coefficients/intercept; load restores exact fitted state for direct `predict_proba()`.

Run:

```powershell
python train_model.py
```

## Exercise 2 — Flask API (`app.py`)

- `GET /` returns usage hint JSON.
- `POST /predict` accepts JSON `{study_hours, attendance_percentage}`.
- Validates: body must be JSON, both fields required and numeric, `study_hours >= 0`, `0 <= attendance <= 100` else `400`.
- Returns `{study_hours, attendance_percentage, prediction (0/1), result (PASS/FAIL), probability_pass}`.
- Uses both features, so attendance influences the prediction (retrained model).

Run:

```powershell
python app.py
```

Test (PowerShell — `Invoke-RestMethod` avoids `curl.exe` quoting issues):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours":5,"attendance_percentage":75}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours":5,"attendance_percentage":150}'
```

First returns `PASS`; second returns `400 attendance must be between 0 and 100`. `GET /predict` correctly returns `405`.

## Exercise 3 — Streamlit Interface (`streamlit_app.py`)

- Inputs: `text_input` student name, `number_input` study hours, `slider` attendance.
- `button("Predict")` loads `student_pass_model.joblib`, runs `predict_proba`, shows `success/info` result message with name.
- `caption` explains generation: saved scaler transform + logistic regression + 0.5 threshold, no retraining.
- Bonus: `bar_chart` sweeping hours 0-10 at current attendance, plotting `P(pass)` and `predicted_class`.

Run:

```powershell
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, e.g. `James / 4h / 75%` -> `PASS (0.96)`.
