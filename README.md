# Machine Learning Exercises 1–3: Student Pass Prediction

This combined project contains:

1. **Exercise 1 — Model Persistence:** train and save a model with `joblib.dump()`, load it with `joblib.load()`, and make three sample predictions.
2. **Exercise 2 — Flask API:** accept study hours and attendance percentage, validate the attendance range, and return a JSON prediction.
3. **Exercise 3 — Streamlit Interface:** enter a student name and study hours, click Predict, and see the result, explanation, and bonus chart.

## Requirements

- Python 3.10 or newer recommended
- Internet access for installing the Python packages (the project itself runs locally after installation)

## Setup

Open a terminal in this project folder and run:

### Windows (PowerShell or CMD)

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python train_model.py
```

If `python` is not recognized, try `py` in place of `python`.

## Exercise 1 — Model Persistence

Run:

```bash
python train_model.py
```

This trains the model from `data/student_data.csv`, saves it as `student_pass_model.joblib`, loads the saved file, and prints predictions for 2, 5, and 8 study hours.

**Why does the model not need to be trained again?** The joblib file stores the fitted model, including its learned parameters. Loading it restores that trained model. Train it again only when the training data or model configuration needs to be updated.

## Exercise 2 — Flask API

Start the API:

```bash
python app.py
```

The API runs at `http://127.0.0.1:5000`.

Send a `POST` request to `http://127.0.0.1:5000/predict` with `Content-Type: application/json`.

### Test with curl.exe (Windows)

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

Example invalid attendance test:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":120}"
```

The invalid request should return HTTP 400 and an error message. You can also test the same endpoint in Postman by choosing **POST**, entering the URL, selecting **Body → raw → JSON**, and pasting:

```json
{
  "study_hours": 5,
  "attendance_percentage": 90
}
```

**Important limitation:** the sample model was trained with `study_hours` only. The API validates and returns `attendance_percentage`, but attendance does not influence the prediction. To make it influence predictions, add an `attendance_percentage` column to the training dataset, update `train_model.py` to train using both features, and retrain the model. Use an appropriate, real dataset for meaningful results.

## Exercise 3 — Streamlit Interface

In another terminal, activate the virtual environment and run:

```bash
streamlit run streamlit_app.py
```

Streamlit opens the interface in your browser. Enter a student name and study hours, then click **Predict**. The page shows the predicted class and an explanation of how the model produced it.

The bonus chart compares the entered study-hours value with the predicted class (0 or 1). These values use different scales, so interpret the chart as a simple input/output illustration rather than a performance evaluation.

## Dataset and interpretation

The included CSV is a small **illustrative practice dataset**, not real student records and not a validated academic prediction system. Its labels are simplified for learning. Do not use this sample model to make real academic decisions.

- `passed = 0`: Needs Improvement
- `passed = 1`: Pass

## Suggested submission checklist

- [ ] `train_model.py` saves and loads `student_pass_model.joblib`.
- [ ] Three predictions are displayed for different study-hour values.
- [ ] Flask returns both inputs and a prediction in JSON.
- [ ] Attendance values outside 0–100 are rejected.
- [ ] Streamlit accepts a name and study hours and displays a result.
- [ ] Include screenshots of successful and invalid API requests.
- [ ] Explain the difference between validating an input and using it as a model feature.
