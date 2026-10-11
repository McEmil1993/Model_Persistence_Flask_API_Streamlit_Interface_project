# Student Pass Prediction

A small machine-learning project that predicts whether a student passes based
on daily study hours. It contains a training script, a Flask API, and a
Streamlit interface.

## Files

| File | Description |
| --- | --- |
| `train_model.py` | Trains a Logistic Regression model, saves it with `joblib.dump()`, reloads it with `joblib.load()`, and makes three sample predictions. |
| `app.py` | Flask API. Accepts `study_hours` and `attendance_percentage`, validates both, and returns the prediction as JSON. |
| `streamlit_app.py` | Streamlit UI with a student name input, study-hours input, prediction button, result message, explanation, and a bonus chart. |
| `student_pass_model.joblib` | The saved (persisted) trained model used by the API and the UI. |

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
# 1. Train and persist the model
python train_model.py

# 2. Start the Flask API (default port 5000)
python app.py
# If port 5000 is busy:  $env:PORT=5001; python app.py

# 3. Start the Streamlit UI
python -m streamlit run streamlit_app.py
```

## API usage

`POST /predict` with a JSON body:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d '{\"study_hours\": 5, \"attendance_percentage\": 90}'
```

Example response:

```json
{
  "study_hours": 5.0,
  "attendance_percentage": 90.0,
  "prediction": 1,
  "result": "PASS",
  "note": "Prediction is based on study hours only. Attendance does not influence the result unless the model is retrained with attendance as an input feature."
}
```

`attendance_percentage` must be between 0 and 100, otherwise the API responds
with HTTP 400.

## Why the model is not retrained for every prediction

Training finds the model's parameters (coefficient and intercept) that best fit
the training data. `joblib.dump()` saves those learned parameters to
`student_pass_model.joblib`, and `joblib.load()` restores them. Making a
prediction only applies the stored formula, so a trained model can be reused
indefinitely. Retraining is only needed when new training data or new features
(such as attendance) are added.
