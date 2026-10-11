# Jay Mark Renos — Student Pass Prediction

This project completes the three exercises in `Exercise.html`:
1. Model persistence with `joblib.dump()` and `joblib.load()`, including three example predictions.
2. Flask API accepting `study_hours` and `attendance_percentage`, validating attendance from 0 to 100, and returning JSON.
3. Streamlit interface with student name, study-hours input, prediction button, result explanation, and chart.

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the first installation of Python packages

## Setup (Windows)
Open Command Prompt in this folder and run:

```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

A sample `student_pass_model.joblib` is included, so you can run the apps after installing the packages. Run `py train_model.py` whenever you want to retrain the model or regenerate the saved model file.

## Run the Streamlit app
```bat
streamlit run streamlit_app.py
```
Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Run the Flask API
Open another Command Prompt in the same folder, activate the environment, then run:

```bat
.venv\Scripts\activate
py app.py
```

The API runs at `http://127.0.0.1:5000`.

### Test with curl.exe
In another Command Prompt, run:

```bat
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

Expected response structure (the prediction and probability depend on the trained model):
```json
{
  "study_hours": 5.0,
  "attendance_percentage": 90.0,
  "prediction": "Pass",
  "pass_probability": 0.8,
  "note": "Prediction uses study_hours only. Attendance is validated and returned but is not used by this model."
}
```

You can also test invalid attendance:
```bat
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":120}"
```
It should return HTTP 400 with an error message.

## Important note about attendance
As stated in the exercise, the provided model currently uses only study hours. The API checks attendance and includes it in the JSON response, but attendance does not influence the prediction. To make attendance affect predictions, update the training dataset and train the model with two input features.

## Model persistence explanation
The model is saved in `student_pass_model.joblib` using `joblib.dump()`. Later, `joblib.load()` restores the learned model from that file. The model does not need to be trained again for each prediction because the learned parameters are already stored in the saved file.

## About the sample dataset
The training values are illustrative classroom data, not real student records. Replace them with the dataset supplied by your instructor if one exists.
