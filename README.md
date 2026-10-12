# Student Prediction Exercise

This project provides example solutions for the three tasks in `Exercise.html`.

## Files

- `train_model.py` — trains a simple example classifier, saves it as `student_pass_model.joblib`, reloads it, and makes three predictions.
- `app.py` — Flask API that accepts `study_hours` and `attendance_percentage`, validates both values, and returns JSON.
- `streamlit_app.py` — simple Streamlit interface with student name, study-hours input, prediction button, explanation, and a chart.
- `requirements.txt` — Python dependencies.

## Setup

Use Python 3.10 or newer. Open a terminal in this folder and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train and save the model:

```bash
python train_model.py
```

This creates `student_pass_model.joblib` in the same folder.

## Run the Flask API

```bash
python app.py
```

Send a request using `curl.exe` in another Windows terminal:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

Attendance must be between 0 and 100. Example response:

```json
{
  "study_hours": 5.0,
  "attendance_percentage": 90.0,
  "prediction": "Pass",
  "note": "Attendance is validated and returned but does not influence this model's prediction."
}
```

## Run the Streamlit interface

```bash
streamlit run streamlit_app.py
```

## Important note

The exercise says the provided model currently uses only study hours. This example preserves that behavior: the Flask endpoint validates and returns attendance, but prediction is based only on study hours. The included training data is illustrative because the uploaded exercise sheet did not include a separate dataset or starter Python files. Replace the sample training data in `train_model.py` with the class-provided data if required. Replace the placeholder name in the header with your own name before submitting.
