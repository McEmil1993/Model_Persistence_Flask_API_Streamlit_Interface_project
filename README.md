# Student Pass Predictor

## Setup
```
pip install -r requirements.txt
python train_model.py
```
This creates `students.csv`, `student_pass_model.joblib` (study hours only) and
`student_pass_model_attendance.joblib` (study hours + attendance).

## Exercise 1 - Model persistence
```
python predict_saved_model.py
```
Loads `student_pass_model.joblib` with `joblib.load()` and predicts for 1, 5 and 9 study hours.

**Why no retraining?** `joblib.dump()` saves the fitted model (its learned coefficients
and scaler) to disk. `joblib.load()` restores that exact state, so `predict()` just
applies the learned parameters to new inputs. Training is only needed when the data
or the features change.

## Exercise 2 - Flask API
Retrained model: `student_pass_model_attendance.joblib` uses both features.
```
python app.py
```
curl (Windows `curl.exe`; use `\"` quoting in PowerShell):
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"
```
Example response:
```
{"attendance_percentage": 85.0, "pass_probability": 0.9663, "predicted_class": 1, "prediction": "pass", "study_hours": 6.0}
```
Attendance outside 0-100 returns HTTP 400 with an error message. Missing or
non-numeric values also return 400.

## Exercise 3 - Streamlit
```
streamlit run streamlit_app.py
```
Includes name input, study-hours input, Predict button, result message, a short
explanation, and the bonus charts (pass probability vs study hours, and the
predicted class for the entered hours).
