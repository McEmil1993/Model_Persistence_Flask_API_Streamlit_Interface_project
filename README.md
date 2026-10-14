# Student Pass Prediction

A small machine-learning demonstration that persists a trained decision-tree model, serves predictions through a Flask API, and provides a Streamlit interface.

## Project contents

| File | Purpose |
| --- | --- |
| `train_model.py` | Trains the model, saves it as `student_pass_model.joblib`, reloads it, and prints sample predictions. |
| `student_pass_model.joblib` | Serialized trained model used by the API. |
| `app.py` | Flask API with a JSON prediction endpoint. |
| `streamlit_app.py` | Web interface for submitting student details and viewing a prediction. |
| `requirements.txt` | Python packages required by the project. |

## Requirements

- Python 3
- `pip`

## Installation

Open a terminal in the project directory and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Train and persist the model

Run:

```bash
python train_model.py
```

The script trains a `DecisionTreeClassifier` using example data where study hours are the only feature. It writes the fitted model to `student_pass_model.joblib`, loads the saved model with `joblib.load()`, and predicts results for three study-hour values.

Predictions do not require fitting the model again: the saved file contains the trained model, and loading it restores that fitted model for inference. Retrain only when you change the training data, features, or model setup.

## Start the Flask API

Run this in one terminal:

```bash
python app.py
```

The API listens at `http://127.0.0.1:5000`. A `GET /` request returns a message indicating the API is running.

### `POST /predict`

Send JSON containing both required fields:

```json
{
  "study_hours": 5,
  "attendance_percentage": 90
}
```

Example using `curl.exe` on Windows:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"
```

Successful responses include the supplied values, the `Pass` or `Fail` prediction, and a note about which feature the model uses. Invalid JSON, missing values, non-numeric values, negative study hours, or attendance outside 0–100 produce an error response with HTTP 400.

**Model limitation:** `attendance_percentage` is validated and returned, but the current classifier was trained with study hours only. Attendance therefore has no effect on the prediction. To use attendance as a model feature, update the training data and feature matrix in `train_model.py`, retrain the model, and keep the prediction input shape in `app.py` consistent with the trained features.

## Start the Streamlit interface

Keep the Flask API running, then open another terminal in the project directory and run:

```bash
streamlit run streamlit_app.py
```

Enter a student name, study hours, and attendance percentage, then select **Predict**. The interface displays the prediction and explains that it is based on study hours. The name is shown in the interface and is not sent to the API.

By default, Streamlit calls `http://127.0.0.1:5000/predict`. To use another API address, set `API_URL` before starting Streamlit. For example, in PowerShell:

```powershell
$env:API_URL = "http://127.0.0.1:5000/predict"
streamlit run streamlit_app.py
```

## Example workflow

1. Install dependencies.
2. Run `python train_model.py` to create or refresh the model file.
3. Start the Flask service with `python app.py`.
4. In another terminal, start the UI with `streamlit run streamlit_app.py`.
5. Submit a request through the UI or test `POST /predict` with `curl.exe`.
