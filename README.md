# Student Pass Predictor

This project implements the three exercises in `../Exercise.html`:

- `train_model.py` trains a model and saves it as `student_pass_model.joblib`.
- `app.py` exposes `/predict` as a Flask JSON API.
- `streamlit_app.py` provides a simple user interface with a prediction chart.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train_model.py
```

The saved model is loaded for predictions, so the applications do not retrain it each time.

## Run the Flask API

```bash
python app.py
```

Test it with:

```bash
curl -X POST http://127.0.0.1:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"study_hours": 5, "attendance_percentage": 85}'
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```
