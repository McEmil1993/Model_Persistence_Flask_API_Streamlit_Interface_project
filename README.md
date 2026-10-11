# IT Elective 3 (AI: Machine Learning) – Final Exercises

**Student:** Kim C. Ferrer
**Course:** IT Elective 3 – Artificial Intelligence: Machine Learning
**Topic:** Model Persistence and Basic Deployment with Flask and Streamlit

---

## About

This repository contains my solutions to the final exercises for IT Elective 3. The exercises use a simple **Logistic Regression** model that predicts whether a student will **Pass or Fail** based on study hours (and, in Exercise 2, attendance percentage).

| Exercise | Description | File |
|----------|-------------|------|
| 1 | Save and load a trained model with `joblib`, then make three predictions | `train_model.py` |
| 2 | Flask API that accepts `study_hours` and `attendance_percentage`, with input validation | `app.py` |
| 3 | Streamlit interface with a name input, Predict button, result message, and chart | `streamlit_app.py` |

## Project Structure

```
.
├── train_model.py                         # Exercise 1: train, save, load, predict
├── app.py                                 # Exercise 2: Flask API
├── streamlit_app.py                       # Exercise 3: Streamlit app
├── student_pass_model.joblib              # Saved model (study hours only)
├── student_pass_model_attendance.joblib   # Saved model (hours + attendance)
└── README.md
```

## Requirements

- Python 3.9 or newer
- scikit-learn, joblib, pandas, Flask, Streamlit

Install the dependencies:

```bash
pip install scikit-learn joblib pandas flask streamlit
```

## How to Run

### Exercise 1 – Save and Load the Model

```bash
python train_model.py
```

This trains the models, saves them as `.joblib` files, loads one back, and prints three predictions:

```
1 study hours -> Fail
4 study hours -> Fail
8 study hours -> Pass
```

**Why no retraining is needed:** the learned parameters (coefficients and intercept) are stored inside the `.joblib` file. Loading it restores the fitted model, so `predict()` only applies those saved values to the new input.

### Exercise 2 – Flask API

Run `train_model.py` first so the model files exist, then start the server:

```bash
python app.py
```

The API runs at `http://127.0.0.1:5000`.

**Endpoint:** `POST /predict`

**Request body:**

```json
{
  "study_hours": 6,
  "attendance_percentage": 85
}
```

**Success response (200):**

```json
{
  "attendance_percentage": 85.0,
  "prediction": "Pass",
  "study_hours": 6.0
}
```

**Validation (400):**

- Missing or non-numeric values return `study_hours and attendance_percentage are required numbers`
- `study_hours` below 0 returns `study_hours must be 0 or greater`
- `attendance_percentage` outside 0–100 returns `attendance_percentage must be between 0 and 100`

**Testing on Windows PowerShell:** save the JSON to a file to avoid quote problems.

```powershell
Set-Content test_ok.json '{"study_hours": 6, "attendance_percentage": 85}'
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "@test_ok.json"
```

**Testing on macOS / Linux:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 6, "attendance_percentage": 85}'
```

### Exercise 3 – Streamlit App

```bash
streamlit run streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`. Enter a student name and study hours, then click **Predict** to see the Pass/Fail result, a short explanation of how it was generated, and a chart of pass probability across study hours.

## Notes

- Everything runs locally, so no internet connection is needed after installing the packages.
- The training data is a small sample dataset written for these exercises.
- If you see a scikit-learn version warning when loading the models, re-run `python train_model.py` to regenerate the `.joblib` files with your installed version.

## Author

**Kim C. Ferrer**
IT Elective 3 – Final Exercises
