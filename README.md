# 🎓 Student Pass Predictor

ML exercises: model persistence (`joblib`), Flask API, and Streamlit UI.
Predicts student Pass/Fail from **study_hours** + **attendance_percentage**
using a `LogisticRegression` model.

## Project structure

```
├── train_model.py            # Ex 1+2: train, joblib.dump, verify via joblib.load
├── student_data.csv          # training data (22 rows)
├── student_pass_model.joblib # trained model (committed deliverable)
├── app.py                    # Ex 2: Flask POST /predict API
├── streamlit_app.py          # Ex 3: Streamlit UI + chart
└── requirements.txt          # pinned dependencies
```

## Setup

```bash
pip install -r requirements.txt
python train_model.py
```

## Exercise 1 — Model persistence

```bash
python train_model.py
```

Trains and saves via `joblib.dump(model, 'student_pass_model.joblib')`,
then reloads via `joblib.load(...)` and prints 3 predictions
(2.5 → Fail, 5.0 → Pass, 8.5 → Pass at 75% attendance).

> Why no retraining? `fit()` learns weights (`coef_`, `intercept_`).
> `joblib.dump` serializes the fitted estimator to disk; `joblib.load`
> restores it with weights intact, so `predict()` is just math — no
> optimization or training data needed. See docstring in `train_model.py`.

## Exercise 2 — Flask API

```bash
python app.py
# http://127.0.0.1:5000
```

Valid:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 6.5, "attendance_percentage": 85.0}'
# {"study_hours":6.5,"attendance_percentage":85.0,"prediction":"Pass"}
```

Invalid attendance (400):
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 6.5, "attendance_percentage": 150}'
```

PowerShell alternative:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST `
  -ContentType "application/json" `
  -Body '{"study_hours":6.5,"attendance_percentage":85.0}'
```

## Exercise 3 — Streamlit UI

```bash
streamlit run streamlit_app.py
# http://localhost:8501
```

Enter student name, study hours, attendance → **Predict Result** →
personalized message + weight explanation + threshold chart.

## Requirements

Pinned, verified working (Python 3.14.6):
`flask==3.1.3 streamlit==1.66.0 joblib==1.6.0 scikit-learn==1.9.1
pandas==3.0.6 matplotlib==3.11.2 plotly==7.1.0 numpy==2.5.4 scipy==1.18.1`
