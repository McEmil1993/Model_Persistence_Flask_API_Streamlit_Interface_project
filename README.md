# Student Outcome Prediction Exercise

This project predicts a sample student's pass class from **study hours per week**
and **attendance percentage**. The included CSV is synthetic classroom data for
practice; it should not be treated as a real academic assessment model.

## Setup

From this directory, install dependencies and fit the model:

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

Training writes `student_pass_model.joblib`. It contains the fitted feature
scaler and logistic-regression classifier. Applications load that file for
predictions, so they do not fit the model for every request. Run training again
if you change the CSV or model design.

## Exercise 1: save, load, and predict

`train_model.py` trains on both CSV columns, persists with `joblib.dump()`, then
restores the fitted pipeline with `joblib.load()`. It predicts for three study
hour values while holding attendance at 80%. The printed explanation describes
why inference can reuse the saved learned parameters without retraining.

## Exercise 2: Flask API

Start the API in one PowerShell window:

```powershell
python app.py
```

In another window, send a request using `curl.exe` (not PowerShell's `curl` alias):

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 20, \"attendance_percentage\": 85}"
```

The response includes both inputs, `predicted_class` (`1` for pass and `0` for
needs support), and a readable `prediction`. Attendance outside 0–100, missing
fields, and invalid JSON data receive a 400 response. Negative study hours are
also rejected. The model uses both validated values.

An invalid attendance request should return HTTP 400:

```powershell
curl.exe -i -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 20, \"attendance_percentage\": 120}"
```

## Exercise 3: Streamlit interface

Run the interface after training:

```powershell
python -m streamlit run streamlit_app.py
```

Enter a student name, study hours, and attendance, then select **Predict
outcome**. The app shows the class and explains which inputs the saved model
used. The chart varies study hours from 0 to 45 while holding the entered
attendance percentage constant, and marks the student's prediction.

## Why a saved model can predict without retraining

Training estimates the model's parameters from labeled examples. `joblib.dump()`
saves those learned parameters (including the scaler), and `joblib.load()` brings
them back into memory. Prediction applies those stored parameters to new feature
values. Retraining is needed when the training examples or model setup changes,
not for each new student.
