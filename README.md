# ML Exercises: Model Persistence, Flask API, Streamlit

## Setup
```
pip install -r requirements.txt
```

## Exercise 1 - Model Persistence
```
python train_model.py
```
Trains a model on study hours, saves it to `student_pass_model.joblib` with
`joblib.dump()`, loads it with `joblib.load()`, and makes three predictions
(1.0, 4.5 and 8.0 hours). The script prints why no retraining is needed:
a fitted model stores its learned parameters, and predicting only applies them
to new input.

## Exercise 2 - Flask API (study hours + attendance)
The original model only knew study hours, so `train_model_v2.py` retrains a
two-feature model (`student_pass_model_v2.joblib`) so attendance really
influences the result.

```
python train_model_v2.py
python app.py
```

Test with curl (Windows: use `curl.exe` and escape quotes, or use Postman):
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"
```
Example response:
```json
{
  "attendance_percentage": 85.0,
  "pass_probability": 0.9937,
  "prediction": "pass",
  "prediction_label": 1,
  "study_hours": 6.0
}
```
Attendance outside 0-100 returns HTTP 400:
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 150}"
```
```json
{"error": "'attendance_percentage' must be between 0 and 100"}
```

## Exercise 3 - Streamlit Interface
Requires `student_pass_model.joblib` (run `python train_model.py` first).
```
streamlit run streamlit_app.py
```
Includes a student name input, study-hours input, Predict button, result
message, a short explanation of how the prediction was made, and a bonus chart
of study hours vs pass probability / predicted class.

## Files
| File | Purpose |
|---|---|
| `train_model.py` | Exercise 1: train, save, load, predict |
| `train_model_v2.py` | Retrains with study hours + attendance |
| `app.py` | Exercise 2: Flask API |
| `streamlit_app.py` | Exercise 3: Streamlit UI |
| `student_pass_model.joblib` | Saved one-feature model |
| `student_pass_model_v2.joblib` | Saved two-feature model |
