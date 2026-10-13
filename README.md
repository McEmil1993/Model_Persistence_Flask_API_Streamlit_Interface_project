# Student Pass Exercises

2-feature `LogisticRegression` pipeline (`study_hours`, `attendance_percentage` -> pass/fail) covering model persistence, Flask API, and Streamlit UI.

## Setup

```powershell
cd C:\Users\ACER\Hnylt_Exercise
pip install scikit-learn joblib flask streamlit pandas numpy altair
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

Expected output:

```text
Training accuracy: 0.952
Test accuracy: 0.922
Model saved to student_pass_model.joblib
Model loaded from student_pass_model.joblib
Study hours=2.0, Attendance=60.0% -> FAIL (P(pass)=0.41)
Study hours=5.0, Attendance=75.0% -> PASS (P(pass)=0.99)
Study hours=8.0, Attendance=90.0% -> PASS (P(pass)=1.00)
```

## Exercise 2 — Flask API (`app.py`)

- `GET /` returns usage hint JSON.
- `POST /predict` accepts JSON `{study_hours, attendance_percentage}`.
- Validates: body must be JSON, both fields required and numeric, `study_hours >= 0`, `0 <= attendance <= 100` else `400`.
- Returns `{study_hours, attendance_percentage, prediction (0/1), result (PASS/FAIL), probability_pass}`.
- Uses both features, so attendance influences the prediction (retrained 2-feature model).

Run (Terminal 1, blocks):

```powershell
python app.py
```

Test (Terminal 2, PowerShell):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/ -Method Get
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours":5,"attendance_percentage":75}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"study_hours":5,"attendance_percentage":150}'
```

Or with `curl.exe` (note `curl.exe`, not bare `curl`):

```powershell
curl.exe http://127.0.0.1:5000/
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":75}"
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":150}"
```

First returns `PASS (0.9865)`; second returns `400 attendance must be between 0 and 100`. `GET /predict` returns `405`.

## Exercise 3 — Streamlit Interface (`streamlit_app.py`)

- Inputs: `text_input` student name, `slider` study hours (0-12), `number_input` attendance (0-100).
- `button("Run prediction")` loads `student_pass_model.joblib`, runs `predict_proba`, shows `metric` + `success`/`error` result message with name.
- `expander` explains generation: saved scaler transform + logistic regression + 0.5 threshold, no retraining.
- Bonus: `bar_chart` sweeping hours 0-10 step 0.5 at current attendance, plotting `P(pass)` and `predicted_class` (x=`study_hours`).

Run:

```powershell
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, e.g. `4.5h / 80%` -> `PASS (0.98)`, sweep `19 of 21` hour steps predict PASS.
