# Student Pass Predictor

A small machine learning project that predicts whether a student will pass or fail.

- **Exercise 1:** Train, save, and load a model with `joblib`
- **Exercise 2:** Flask API that accepts study hours and attendance percentage
- **Exercise 3:** Streamlit web interface

## Project files

| File | Purpose |
|---|---|
| `train_model.py` | Trains and saves the models, then makes three test predictions |
| `app.py` | Flask API (`/predict` endpoint) |
| `streamlit_app.py` | Streamlit web interface |
| `student_pass_model.joblib` | Study-hours model (created by `train_model.py`) |
| `student_pass_model_v2.joblib` | Hours + attendance model (created by `train_model.py`) |

## Requirements

- Python 3.13 or 3.14 (recommended). Make sure **Add python.exe to PATH** was ticked during installation.
- A terminal opened inside the project folder (in VS Code: **Ctrl + `**)

Check Python:

```
python --version
```

## 1. Install the packages

```
python -m pip install scikit-learn flask joblib pandas streamlit
```

## 2. Train and save the models

Run this first. The other two programs need the `.joblib` files it creates.

```
python train_model.py
```

Expected output:

```
Model saved to student_pass_model.joblib
Model loaded from student_pass_model.joblib
Study hours = 2: prediction = FAIL (pass probability = 0.05)
Study hours = 5: prediction = PASS (pass probability = 0.65)
Study hours = 8: prediction = PASS (pass probability = 0.98)
Model v2 (hours + attendance) saved to student_pass_model_v2.joblib
```

**Why no retraining is needed:** `fit()` learns the model's values from data, and `joblib.dump()` saves them to disk. `joblib.load()` restores the same trained model, so predictions only apply the stored values to new input.

## 3. Run the Flask API

```
python app.py
```

The API runs at `http://localhost:5000`. Keep this terminal open and use a **second terminal** for testing.

### Test with curl (Windows PowerShell)

Valid request:

```
curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":7,\"attendance_percentage\":85}"
```

Response:

```json
{"attendance_percentage":85.0,"prediction":"PASS","study_hours":7.0}
```

Invalid attendance (over 100):

```
curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":7,\"attendance_percentage\":120}"
```

Response (HTTP 400):

```json
{"error":"attendance_percentage must be between 0 and 100"}
```

### Test with Postman

1. Set the method to **POST** and the URL to `http://localhost:5000/predict`.
2. Open **Body → raw → JSON** and enter:
   ```json
   {"study_hours": 7, "attendance_percentage": 85}
   ```
3. Click **Send**.

Press **Ctrl + C** in the terminal to stop the API.

## 4. Run the Streamlit app

```
python -m streamlit run streamlit_app.py
```

- If Streamlit asks for an email, press **Enter** to skip it.
- The app opens in your browser at `http://localhost:8501`.
- Enter a student name and study hours, then click **Predict**.
- Press **Ctrl + C** in the terminal to stop it.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Python was not found` | Install Python from python.org, tick **Add python.exe to PATH**, and restart VS Code. Also turn off the Store shortcut in **Settings → Apps → Advanced app settings → App execution aliases**. |
| `streamlit` is not recognized | Use `python -m streamlit run streamlit_app.py` instead. |
| `No module named ...` | Run the install command in step 1 again. |
| `FileNotFoundError: ... .joblib` | Run `python train_model.py` first, and make sure the terminal is inside the project folder. |
| `Failed building wheel` during install | Your Python version is too new for the packages. Use Python 3.13 or 3.14. |
| Port already in use | Close the other terminal running the app, or restart VS Code. |