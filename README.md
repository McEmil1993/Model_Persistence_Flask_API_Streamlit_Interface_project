# Student Pass Prediction Activity

This example implements the model-persistence, Flask API, and Streamlit
exercises. The small training dataset in [train_model.py](./train_model.py) is illustrative only;
it is not real student data and should not be used to make actual academic
decisions.

## Setup

Open Git Bash in the project folder (`/c/MY_AllFolder/Bryan`) and run:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

If `python` is not recognized in Git Bash, try `py` instead when creating the environment, or use the Python executable installed on your system.

## Exercise 1: save and reuse a model

Train, save, reload, and predict with the model:

```bash
python train_model.py
```

The script learns a study-hours decision threshold, saves it as `student_pass_model.joblib` with `joblib.dump()`, then reloads it with `joblib.load()` and predicts results for 2, 5, and 8 study hours. Training learns the model's decision rule; the saved file preserves that rule, so predictions can be made by loading the model without fitting it again.

## Exercise 2: Flask API

Start the API in a Git Bash window after training the model:

```bash
python app.py
```

The API listens at `http://127.0.0.1:5000`. In a second Git Bash window, send a request with `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours":6,"attendance_percentage":92}'
```

The JSON response includes both inputs and the predicted class. Attendance is validated from 0 through 100, but it is not a model feature, so changing attendance alone does not affect the prediction. For example, try an invalid attendance value:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours":6,"attendance_percentage":120}'
```

The invalid request returns HTTP 400 with a validation error.

## Exercise 3: Streamlit interface

In another Git Bash window, run:

```bash
streamlit run streamlit_app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`; open it in your browser. Enter a name and study hours, then select **Predict**. The page shows the predicted result, explains that the saved model uses study hours only, and plots predicted class (0 = Fail, 1 = Pass) against study hours.

## Using attendance in predictions

The API accepts and returns attendance for the exercise, but the provided model is trained on study hours only. To make attendance affect predictions, include attendance as a feature in a real training dataset, retrain the model, and update the prediction input to provide both features. Do not treat the illustrative data in this example as a reliable predictor.


cd /c/MY_AllFolder/Bryan
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run streamlit_app.py
  