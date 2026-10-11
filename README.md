# Student Pass Prediction — Combined Exercises

This project combines all three instructor exercises:

1. **Model Persistence:** `train_model.py` trains a sample model, saves it as `student_pass_model.joblib` with `joblib.dump()`, reloads it with `joblib.load()`, and makes three predictions.
2. **Flask API:** `app.py` accepts `study_hours` and `attendance_percentage`, validates both, and returns JSON containing both inputs and the prediction.
3. **Streamlit Interface:** `streamlit_app.py` collects a student name, study hours, and attendance, displays the result and explanation, and includes the optional chart.

**Attendance note:** This sample model is trained using study hours only, following the instructor's note. Attendance is validated and returned by the API, but does not affect the prediction unless the training data and model are updated to use attendance and the model is retrained. The small dataset is example data for learning, not a reliable real-world predictor.

## Setup on Windows

Open a terminal in this project folder.

### 1. Create a virtual environment (recommended)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Exercise 1 — Train/save and test the model

```powershell
python train_model.py
```

This creates `student_pass_model.joblib` and prints predictions for 2, 5, and 8 study hours. The model file is generated locally by this script.

### 4. Exercise 2 — Start Flask

```powershell
python app.py
```

Keep this terminal open. The API runs at `http://127.0.0.1:5000`.

### 5. Test in Postman (required)

1. Open Postman and create a new HTTP request.
2. Select method **POST**.
3. Enter `http://127.0.0.1:5000/predict`.
4. Select **Body** → **raw** → choose **JSON** from the format dropdown.
5. Paste this request body:

```json
{
  "study_hours": 5,
  "attendance_percentage": 90
}
```

6. Click **Send**. A successful response includes both values, the prediction, pass probability, and a note about attendance.

**Test validation:** Change `attendance_percentage` to `120` and click Send. The API should return **400 Bad Request** and an error explaining that attendance must be between 0 and 100.

### 6. Exercise 3 — Start Streamlit

Open a second terminal in the same project folder. Activate the virtual environment in that terminal too, then run:

```powershell
streamlit run streamlit_app.py
```

Streamlit opens a local webpage. Keep Flask running in the first terminal because the Streamlit interface sends requests to the Flask API.

## Why don't we need to train again for each prediction?

The model's learned parameters are saved in `student_pass_model.joblib`. Loading this file restores the trained model so it can make predictions on new input. Retraining is needed when the training data, features, or model needs to change—not for every prediction.

## Files

- `train_model.py` — trains, saves, reloads, and tests the model.
- `app.py` — Flask API tested with Postman and used by Streamlit.
- `streamlit_app.py` — user interface and bonus chart.
- `requirements.txt` — required Python packages.
- `student_pass_model.joblib` — created after running `python train_model.py`.
