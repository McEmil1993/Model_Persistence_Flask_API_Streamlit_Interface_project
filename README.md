# Student Pass Prediction Project

## Setup
```
pip install -r requirements.txt
```

## Exercise 1 - Model Persistence
```
python train_model.py
```
Trains the model (study hours only), saves it as `student_pass_model.joblib` with
`joblib.dump()`, loads it with `joblib.load()`, and predicts for 1, 5 and 9 study hours.

**Why no retraining is needed:** training is the step that learns the model's
parameters (the coefficients and intercept) from data. `joblib.dump()` stores those
learned parameters in the file. `joblib.load()` restores the exact same fitted model,
so `predict()` only applies the already-learned numbers to the new input. Training
is slow and done once; prediction is fast and can be repeated at any time.

## Exercise 2 - Flask API
Retrain with attendance included, then start the server:
```
python train_model_attendance.py
python app.py
```
Test with curl (Windows: `curl.exe`):
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"
```
Invalid attendance (returns HTTP 400):
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 150}"
```
In Postman: POST to `http://127.0.0.1:5000/predict`, Body -> raw -> JSON.

## Exercise 3 - Streamlit
Run `python train_model.py` first, then:
```
streamlit run streamlit_app.py
```
