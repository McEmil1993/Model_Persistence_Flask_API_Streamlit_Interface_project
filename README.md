# Student Performance Prediction Exercise

This project uses an illustrative sample dataset (`dataset.csv`) with study hours and a binary outcome. The LogisticRegression model is trained on study hours only. Attendance is accepted and validated by the Flask endpoint, then returned in its JSON response; it does not affect model predictions.

## Setup and run

```powershell
py -m pip install -r requirements.txt
py train_model.py
py app.py
```

In another terminal, run the API example below. Start the interface with `py -m streamlit run streamlit_app.py`.

## API example

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":85}"
```

Example response (the class comes from the saved model):

```json
{"attendance_percentage":85.0,"prediction":"Pass","study_hours":5.0}
```

Missing fields, nonnumeric values, and attendance outside 0 through 100 return a JSON `error` and HTTP 400.

## Persistence explanation

Training fits a model and stores learned parameters and patterns from the training data. Saving that fitted model with `joblib` lets a later process load it and predict directly, without repeating training.

## Sample results

Run `py train_model.py` to print predictions at 2, 5, and 8 study hours. The app and API load the same `student_pass_model.joblib` file.
