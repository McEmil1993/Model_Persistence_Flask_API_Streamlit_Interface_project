# Student Pass Prediction

A small demo project that predicts whether a student will **Pass** or **Fail**
based on **study hours** and **attendance percentage**, using logistic regression.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Trains the model on sample data and saves `student_pass_model.joblib` |
| `app.py` | Flask REST API with a `POST /predict` endpoint |
| `streamlit_app.py` | Interactive Streamlit UI with a pass-probability chart |
| `tests/test_app.py` | API tests (run with pytest) |

## Setup

```bash
pip install -r requirements.txt
python train_model.py
```

## Run the API

```bash
python app.py
```

Then call the endpoint:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 6, "attendance_percentage": 85}'
```

Response:

```json
{
  "study_hours": 6.0,
  "attendance_percentage": 85.0,
  "prediction": "Pass"
}
```

Validation errors return HTTP 400 with an `error` message (missing fields,
non-numeric values, attendance outside 0–100, negative study hours).

## Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Tests

```bash
python -m pytest
```

## Notes

- The training data in `train_model.py` is a tiny hand-crafted sample (12 rows).
  Replace it with real data for any serious use.
- `student_pass_model.joblib` is a generated artifact; re-run `train_model.py`
  after changing the training data.
