# Student Pass Prediction — Exercises 1–3

This project demonstrates model persistence with Joblib, a Flask JSON API, and a Streamlit interface.

## 1. Setup

Use Python 3.10 or newer. Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and train/save the model:

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

The training script saves `student_pass_model.joblib`, loads it again with `joblib.load()`, and makes predictions for 2, 5, and 8 study hours.

## Exercise 1 — Model Persistence

- `joblib.dump(model, MODEL_PATH)` saves the fitted model to a file.
- `joblib.load(MODEL_PATH)` restores the fitted model.
- The three sample predictions are made for 2, 5, and 8 study hours.

The model does not need to be trained again for predictions because the saved file contains the fitted model parameters. Load the file and call `predict()`. Retraining is needed when you want to fit the model again with new training data or changed features.

## Exercise 2 — Flask API

Start the API:

```powershell
python app.py
```

Test with `curl.exe` in another PowerShell terminal:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

Example request body for Postman (method `POST`, URL `http://127.0.0.1:5000/predict`, Body → raw → JSON):

```json
{
  "study_hours": 5,
  "attendance_percentage": 90
}
```

The endpoint validates that attendance is from 0 to 100 and returns both inputs plus the prediction as JSON. **Important:** the model currently uses study hours only. Attendance is validated and returned, but it does not affect the prediction. To make attendance affect predictions, train a model using both study hours and attendance percentage as input features.

## Exercise 3 — Streamlit Interface

In a separate terminal, from this same folder, run:

```powershell
streamlit run streamlit_app.py
```

Enter a student name and study hours, then click **Predict result**. The page displays the prediction, an estimated pass probability, an explanation, and a chart.

## Notes

- The included training data is small and illustrative for classroom practice, not a validated predictor of real academic performance.
- Predicted class: `0 = Fail`, `1 = Pass`.
- The API's `attendance_percentage` is not a model feature yet.
