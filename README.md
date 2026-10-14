# Student Pass Prediction — Exercises 1 to 3 (with Attendance)

## Included exercises
1. **Exercise 1 — Model Persistence:** trains a model using both study hours and attendance, saves it as `student_pass_model.joblib`, reloads it, and makes three predictions.
2. **Exercise 2 — Flask API:** `POST /predict` accepts `study_hours` and `attendance_percentage` and returns the prediction as JSON.
3. **Exercise 3 — Streamlit UI:** accepts student name, study hours, and attendance percentage, then displays a prediction.

## Setup
Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python train_model.py
```

The training script creates `student_pass_model.joblib`. Keep it in the same folder as the Python scripts.

## Run Exercise 2 — Flask API
```bash
python app.py
```
API URL: `http://127.0.0.1:5000`

Example request body (POST, Body → raw → JSON):
```json
{
  "study_hours": 5,
  "attendance_percentage": 80
}
```

Example Windows test:
```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":80}"
```

## Run Exercise 3 — Streamlit
```bash
streamlit run streamlit_app.py
```
Enter the student name, study hours, and attendance percentage, then click **Predict result**.

## Exercise 1 explanation
The model is trained with two features: study hours and attendance percentage. `joblib.dump()` saves the trained model; `joblib.load()` restores it so predictions can be made without training again.

## Important note
The training data is illustrative sample data for practice. Predictions are not guaranteed to reflect real student performance. Use suitable real data and evaluate the model before relying on results.
