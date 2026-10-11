# Student Prediction Exercises — Exercises 1, 2, and 3

This folder contains all three exercises in one Python project.

## Files

- `train_model.py` — Exercise 1: trains the example model, saves it as `student_pass_model.joblib`, reloads it, and makes three predictions.
- `app.py` — Exercise 2: Flask API that accepts study hours and attendance percentage, validates attendance (0–100), and returns the inputs and prediction as JSON.
- `streamlit_app.py` — Exercise 3: student-name and study-hours inputs, Predict button, result message, explanation, and bonus chart.
- `requirements.txt` — Python dependencies.
- `student_pass_model.joblib` — Generated when you run `train_model.py`.

## 1. Set up on Windows

Open PowerShell in this project folder.

Check that Python is installed:

```powershell
py --version
```

If that command works, install the packages:

```powershell
py -m pip install -r requirements.txt
```

If `py --version` does not work, install Python from https://www.python.org/downloads/ and select **Add Python to PATH** during installation. You can also disable the Microsoft Store Python execution aliases in Windows Settings if they intercept the command.

## 2. Exercise 1 — Model Persistence

Run:

```powershell
py train_model.py
```

This creates `student_pass_model.joblib`, loads the saved model with `joblib.load()`, and predicts results for 2, 5, and 8 study hours.

**Why is retraining unnecessary for predictions?** The saved file contains the fitted model and its learned decision rules. Loading it restores the trained model. Retraining is only needed when updating the model or learning from new training data.

## 3. Exercise 2 — Flask API

First, run `py train_model.py` if the `.joblib` file does not exist. Then start the API:

```powershell
py app.py
```

The API runs at `http://127.0.0.1:5000`.

### Test using curl.exe

Open another PowerShell window and run:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":6,\"attendance_percentage\":90}"
```

Example response:

```json
{
  "study_hours": 6.0,
  "attendance_percentage": 90.0,
  "predicted_class": 1,
  "prediction": "Pass",
  "note": "Prediction uses study_hours only; attendance is validated but is not a model feature."
}
```

You can test invalid attendance:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":6,\"attendance_percentage\":120}"
```

This should return HTTP 400 and an error message.

### Test using Postman

1. Create a new **POST** request to `http://127.0.0.1:5000/predict`.
2. Choose **Body → raw → JSON**.
3. Enter:

```json
{
  "study_hours": 6,
  "attendance_percentage": 90
}
```

4. Click **Send**. Try attendance values below 0 or above 100 to test validation.

**Important:** The exercise prompt says the supplied model uses only study hours. Therefore, attendance is accepted, validated, and returned, but it does not influence the predicted class. To make attendance affect predictions, add it as a second feature in the training data and retrain the model.

## 4. Exercise 3 — Streamlit Interface

If the model file does not exist, run `py train_model.py` first. Then start Streamlit:

```powershell
py -m streamlit run streamlit_app.py
```

Streamlit will open the app in your browser. Enter a student name and study hours, then click **Predict**. The app displays the predicted class, a short explanation, and a chart.

## Educational-use note

The model is intentionally small and uses illustrative sample data so the exercises can run independently. Its predictions are not a validated measure of real academic performance. Replace the sample data with appropriate real data before using a model for any meaningful assessment.
