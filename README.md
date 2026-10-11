# Student Prediction Exercises

This project contains sample solutions for the three tasks in `Exercise.html`.

## 1. Set up

Use Python 3.10 or newer. Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
```

## Exercise 1 — Model Persistence

Train and save the model, then load it and make three predictions:

```bash
python train_model.py
```

This creates `student_pass_model.joblib`. The sample training data is for demonstration only. Replace it with your teacher's dataset if one was provided.

**Why no retraining is needed:** `joblib.dump()` saves the fitted model and its learned parameters. `joblib.load()` restores those parameters, so the loaded model can make predictions directly. Retrain only when you want to fit it to new or changed training data.

## Exercise 2 — Flask API

Run:

```bash
python app.py
```

Send a POST request to `http://127.0.0.1:5000/predict` with JSON:

```json
{
  "study_hours": 5,
  "attendance_percentage": 90
}
```

Using curl:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 90}"
```

The API validates study hours (0–24) and attendance (0–100). As the exercise notes, the model currently uses study hours only; attendance is returned but does not influence the prediction.

## Exercise 3 — Streamlit Interface

Run:

```bash
streamlit run streamlit_app.py
```

Enter a student name and study hours, then press **Predict result**. The page displays the prediction, a short explanation, and a bonus chart.

## Important note

The example uses made-up demonstration data to make the files runnable because the exercise HTML did not include a separate training dataset or starter Python files. Predictions are for practice only and are not reliable academic assessments.
