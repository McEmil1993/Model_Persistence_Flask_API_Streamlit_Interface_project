# Student Pass Predictor

Three exercises demonstrating model persistence, a Flask API, and a Streamlit
interface, all built around a single logistic regression model that predicts
whether a student will pass based on their daily study hours.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Exercise 1 — generates the dataset, trains and saves the model, then reloads it and makes predictions |
| `app.py` | Exercise 2 — Flask API serving the saved model |
| `streamlit_app.py` | Exercise 3 — Streamlit web interface |
| `requirements.txt` | Dependencies |
| `student_data.csv` | Generated dataset (recreated by `train_model.py`) |
| `student_pass_model.joblib` | Saved model (recreated by `train_model.py`) |

The two generated files are committed for convenience. Delete them and rerun
`train_model.py` to reproduce them exactly — the random seed is fixed.

## Setup

```bash
python -m venv .venv
```

Activate it, then install the dependencies:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

All commands below assume the virtual environment is active.

---

## Exercise 1 — Model Persistence

```bash
python train_model.py
```

**Expected output:**

```
Generated student_data.csv with 100 students.
  Rows where the student passed: 51 / 100
  Attendance column is present but unused by the model (see note below).

Trained model saved to student_pass_model.joblib using joblib.dump().
  Learned coefficient: 1.1590
  Learned intercept:   -3.8255

Loaded model from student_pass_model.joblib using joblib.load().

Making three predictions:
   1.0 study hours -> Fail (probability of passing: 6.5%)
   3.0 study hours -> Fail (probability of passing: 41.4%)
   6.0 study hours -> Pass (probability of passing: 95.8%)
```

The script then prints the answer to the final task — why the model does not
need to be trained again in order to make predictions.

**How to verify persistence:** note the printed coefficient (1.1590), then run
the script again. It is identical every time, because the random seed is fixed.
The three predictions come from a freshly written file that was read back with
`joblib.load()`.

**About the data:** the dataset is synthetic. `study_hours` (hours studied per
day, 0–8) drives the outcome, and `attendance_percentage` is generated and
stored but deliberately *not* used by the model — see the note under
Exercise 2.

---

## Exercise 2 — Flask API

Start the server:

```bash
python app.py
```

Wait for: `Running on http://127.0.0.1:5000`

Leave it running and open a **second terminal** for the tests below.

### Test 1 — valid request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 7, "attendance_percentage": 85}'
```

Expected (HTTP 200):

```json
{
  "attendance_percentage": 85.0,
  "prediction": 1,
  "result": "Pass",
  "study_hours": 7.0
}
```

### Test 2 — validation errors

Each of these must return HTTP 400 with a descriptive message.

```bash
# Attendance outside 0-100
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 7, "attendance_percentage": 150}'
# -> {"error": "'attendance_percentage' must be between 0 and 100."}

# Negative study hours
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": -3, "attendance_percentage": 85}'
# -> {"error": "'study_hours' cannot be negative."}

# Missing field
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 7}'
# -> {"error": "Missing required field: 'attendance_percentage'."}

# Non-numeric values
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": "seven", "attendance_percentage": 85}'
# -> {"error": "'study_hours' and 'attendance_percentage' must be numbers."}
```

Add `-w "\nHTTP %{http_code}\n"` to any command to display the status code.

### Test 3 — attendance does not affect the prediction (by design)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 2, "attendance_percentage": 100}'
```

Result is **Fail** despite 100% attendance. This is intentional and matches the
note on the exercise sheet: the model was trained on `study_hours` only, so
`attendance_percentage` is validated and echoed back in the response but never
reaches the model. Making attendance influence the prediction requires
retraining on both columns — that is the follow-up task the note describes, and
`student_data.csv` already contains the attendance column ready to use.

Stop the server with `Ctrl+C`.

---

## Exercise 3 — Streamlit Interface

```bash
streamlit run streamlit_app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) and normally
opens it in your browser. If it does not, open the URL manually.

**To test:**

1. Enter a student name (optional — leave blank to see the generic message).
2. Set the study hours using the number input.
3. Click **Predict**.

| Student | Study hours | Expected result |
|---|---|---|
| Alice | 1.0 | Fail (~6.5% probability) |
| Bob | 6.0 | Pass (~95.8% probability) |
| (any) | 3.0 | Fail (~41.4% — right at the boundary) |

The app displays:

- The result message, styled differently for pass and fail.
- **How this prediction was generated** — the model's learned coefficient and
  intercept, the arithmetic for the entered study hours, and the resulting
  probability against the 0.5 decision threshold.
- **Study hours vs. predicted class** (bonus) — the probability curve across
  0–8 daily study hours, with your input marked by a red dashed line and the
  0.5 decision boundary drawn as a horizontal line.

Stop the server with `Ctrl+C`.

---

## Implementation notes

**The model.** A logistic regression trained on `study_hours` (hours studied
per day) alone. It learns a coefficient and an intercept; a prediction
multiplies the study hours by the coefficient, adds the intercept, and passes
the result through the logistic function to obtain a probability. Below 0.5
predicts Fail, at or above predicts Pass. The learned decision boundary sits at
roughly 3.3 daily study hours.

**Why persistence matters.** Training is the search for the coefficient and
intercept. `joblib.dump()` writes them to disk and `joblib.load()` restores
them, so the trained model can be served by both the Flask API and the
Streamlit app without retraining. Predicting afterwards is only arithmetic on
the saved parameters.

**File paths.** All three scripts resolve `student_pass_model.joblib` relative
to their own location rather than the current working directory, so they behave
identically no matter which folder they are launched from.

**Missing model file.** If `student_pass_model.joblib` is absent, `app.py`
refuses to start and `streamlit_app.py` displays a message asking you to run
`train_model.py` first. Neither crashes with a stack trace.
