# Student Pass Project

## Setup
```
pip install -r requirements.txt
python train_model.py              # Exercise 1: saves student_pass_model.joblib
python train_model_attendance.py   # Exercise 2: saves student_pass_model_attendance.joblib
```

## Exercise 1 - Model persistence
```
python train_model.py
python predict_demo.py
```
**Why no retraining is needed:** training is the step where the model learns its
parameters (the coefficients and intercept) from data. `joblib.dump()` saves those
learned parameters to disk. `joblib.load()` restores the exact same fitted model, so
`predict()` only plugs the new study hours into the already-learned formula. Training
is slow and done once; predicting is fast and can be repeated as often as needed.

## Exercise 2 - Flask API
```
python app.py
```
curl.exe (Windows PowerShell):
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"
```
macOS/Linux:
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours": 6, "attendance_percentage": 85}'
```
Invalid attendance (returns HTTP 400):
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 130}"
```
In Postman: POST to `http://127.0.0.1:5000/predict`, Body -> raw -> JSON, same payload.

## Exercise 3 - Streamlit
```
streamlit run streamlit_app.py
```
