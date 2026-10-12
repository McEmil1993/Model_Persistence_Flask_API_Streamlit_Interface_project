# Student Pass Prediction Exercise

This project implements the three exercises from the supplied instructions.

## Project files

- `train_model.py` — trains and persists the model with Joblib, then loads it and makes three predictions.
- `student_pass_model.joblib` — saved trained model.
- `app.py` — Flask API with `/predict` endpoint.
- `streamlit_app.py` — Streamlit interface with student name, study hours, prediction button, explanation, and bonus chart.
- `requirements.txt` — Python dependencies.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Train and save the model

```bash
python train_model.py
```

The script saves `student_pass_model.joblib` and then loads that saved file to make predictions for 2, 5, and 8 study hours.

The model does not need to be trained again for those predictions because the trained model parameters have already been serialized to disk with `joblib.dump()` and restored with `joblib.load()`.

## 3. Run the Flask API

```bash
python app.py
```

### Test with curl.exe

```bash
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":6,\"attendance_percentage\":90}"
```

### Invalid attendance test

```bash
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":6,\"attendance_percentage\":120}"
```

The second request returns an error because attendance must be between 0 and 100.

### Important note about attendance

The exercise states that the provided model currently uses only study hours. Therefore, this implementation accepts and validates `attendance_percentage`, returns it in the JSON response, but does **not** use attendance to calculate the prediction. To make attendance influence predictions, the training data, model features, and training process must be updated and the model retrained.

## 4. Run Streamlit

```bash
streamlit run streamlit_app.py
```

The interface provides:

- Student name input
- Study-hours input
- Prediction button
- Pass/Fail result
- Explanation of how the prediction was generated
- Bonus chart comparing the entered study hours with the predicted class
