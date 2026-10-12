# Student Prediction Exercises — Solutions

This folder contains example solutions for all three exercises in `Exercise.html`.

## Files
- `train_model.py` — trains a Logistic Regression classifier, saves it as `student_pass_model.joblib`, reloads it, and prints three predictions.
- `app.py` — Flask API with `POST /predict`, validates study hours and attendance, and returns JSON.
- `streamlit_app.py` — student name and study-hours inputs, prediction button, result message, explanation, and a bonus chart.
- `requirements.txt` — Python dependencies.

## Setup (Windows PowerShell)

```powershell
cd student_prediction_exercises
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
```

Run the Flask API in one terminal:

```powershell
python app.py
```

Test with `curl.exe` in another terminal:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":6,\"attendance_percentage\":90}"
```

Example validation test (attendance is invalid):

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":6,\"attendance_percentage\":120}"
```

Run the Streamlit interface:

```powershell
streamlit run streamlit_app.py
```

## Important model limitation

The provided sample model uses **study hours only**. The Flask endpoint accepts, validates, and returns `attendance_percentage`, but attendance does not influence the prediction. This matches the exercise note. To make attendance affect predictions, obtain suitable training data, train the model with both features, and update the prediction code to pass both features.

The sample training data is illustrative and should not be treated as a validated predictor of real student outcomes.
