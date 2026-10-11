# Model Persistence and Basic Deployment: Exercises

IT Elective 3 (AI: Machine Learning). A model predicts whether a student passes or fails from study hours (and attendance), is saved once, then reused by a Flask API and a Streamlit app.

| File | Purpose |
| --- | --- |
| `train_model.py` | Exercise 1. Trains, saves (`joblib.dump`), loads (`joblib.load`) and predicts |
| `app.py` | Exercise 2. Flask API (`POST /predict`) |
| `streamlit_app.py` | Exercise 3. Streamlit interface |
| `test_api.py` | Automated tests for the API and model persistence |
| `postman_collection.json` | 9 ready-made Postman requests |
| `students.csv` | Sample data (40 made-up students: `study_hours`, `attendance_percentage`, `passed`) |
| `student_pass_model.joblib` | Saved model, study hours only (Exercises 1 and 3) |
| `student_pass_model_attendance.joblib` | Saved model, study hours + attendance (Exercise 2) |

## Setup

Requires Python 3.10 or newer. Using a virtual environment keeps the libraries for this project separate:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

(On macOS or Linux, activate with `source .venv/bin/activate`.) Always run `train_model.py` on your own machine first: a saved model should be loaded with the same scikit-learn version that created it.

---

## Exercise 1: Model Persistence (`train_model.py`)

- Saves the study-hours model as `student_pass_model.joblib` with `joblib.dump()`.
- Loads it back with `joblib.load()` and prints the coefficient and intercept stored in the file.
- Makes three predictions (1, 4 and 8 study hours).
- Checks `students.csv` first and stops with a clear message if a column is missing.

**Why does the model not need to be trained again?**
Training (`.fit()`) is where the algorithm learns its values (the coefficients and intercept) from the data. Those learned values live inside the model object. `joblib.dump()` writes the whole object to a file and `joblib.load()` restores it exactly. `predict()` then just applies the already-learned values to the new input, so it needs neither `.fit()` nor the training data.

---

## Exercise 2: Flask API (`app.py`)

`train_model.py` also trains a second model on study hours **and** attendance, so attendance really affects the prediction. The API uses that model.

```
python app.py
```

**Request:** `POST http://127.0.0.1:5000/predict` with a JSON body

```json
{"study_hours": 6, "attendance_percentage": 90}
```

**Response (200):**
```json
{"attendance_percentage": 90.0, "pass_probability": 0.977, "prediction": "Pass", "study_hours": 6.0}
```

| Status | When |
| --- | --- |
| 200 | Valid request |
| 400 | Missing field, not a number, `study_hours` below 0, `attendance_percentage` outside 0 to 100, or broken JSON. The body is `{"errors": [...]}` |
| 415 | Content-Type is not JSON (in Postman, Body type was left on Text) |
| 404 / 405 | Wrong URL / wrong method (the API answers in JSON) |

### Test with Postman

1. Run `python app.py` and keep it running.
2. In Postman choose **Import** and select `postman_collection.json`. Or make one request yourself: method `POST`, URL `http://127.0.0.1:5000/predict`, **Body > raw > JSON**, and paste the request above.
3. Press **Send** and check the status code and the JSON.

| Request | Expected |
| --- | --- |
| 1. Valid (6 hours, 90%) | 200, `Pass` |
| 2. Valid, low values (1 hour, 60%) | 200, `Fail` |
| 3. Attendance 0 | 200 (boundary is allowed) |
| 4. Attendance 100 | 200 (boundary is allowed) |
| 5. Attendance 120 | 400 |
| 6. Attendance -5 | 400 |
| 7. `"five"` instead of a number | 400 |
| 8. Missing attendance | 400 |
| 9. GET instead of POST | 405 |

Each imported request has a test (see the **Tests** tab) that shows pass or fail after you send it.

---

## Exercise 3: Streamlit Interface (`streamlit_app.py`)

```
streamlit run streamlit_app.py
```

Has a student name input, a study-hours input, a **Predict** button (Enter also works), a result message, and a short explanation of how the prediction was generated, with the real numbers for that student.

Bonus chart: study hours vs. pass probability, with the area where the model predicts **Fail** (red) and **Pass** (green), the past students, and the current student as a blue diamond. If you re-train the model while the app is open, it loads the new model on the next prediction.

---

## Automated tests

With no server running:

```
python -m unittest -v
```

15 tests cover the 0 to 100 attendance rule (including the boundaries), wrong types, missing fields, NaN and huge numbers, the 400/404/405/415 responses, and that a saved and reloaded model gives identical predictions.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `model not found` / `Missing file(s)` | Run `python train_model.py` first |
| `python` is not recognized | Reinstall Python and tick "Add python.exe to PATH", or use `py` |
| Postman: "Could not send request" | `app.py` is not running, or its window was closed |
| Postman returns 415 | Body > raw, then change the dropdown from Text to **JSON** |
| Port 5000 already in use | Close the other window that is running the app |
| Version warning when loading a model | Run `python train_model.py` again on your machine |

## Notes

- **Sample data:** `students.csv` is made up and tiny, so the accuracy numbers are only a rough indicator. To use your own data keep the same column names and run `train_model.py` again.
- **`debug=True`** in `app.py` is for learning only. For a real deployment use a production server, for example on Windows: `pip install waitress`, then `waitress-serve --listen=127.0.0.1:5000 app:app`.
- Tested with Python 3.12, scikit-learn 1.8 and 1.9, pandas 3.0, Flask 3.1 and Streamlit 1.65, including a clean install from `requirements.txt`.
