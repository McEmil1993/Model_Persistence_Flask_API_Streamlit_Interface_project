# Student Pass Prediction Exercises

This project provides sample solutions for the three exercises in `Exercise.html`.

## 1. Set up the project

Use Python 3.10 or newer. Open a terminal in this folder and run:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Exercise 1 — Model persistence

Train and save the model, then load it and make three predictions:

```powershell
python train_model.py
```

The script creates `student_pass_model.joblib`.

**Why does the model not need to be trained again for each prediction?**  
Training learns the model parameters from the example data. `joblib.dump()` saves those learned parameters to disk, and `joblib.load()` restores them. The loaded model can make predictions on new inputs without repeating training, unless the training data/model needs to be updated.

## Exercise 2 — Flask API

First ensure the model exists by running `python train_model.py`. Then start the API:

```powershell
python app.py
```

Send a POST request to `http://127.0.0.1:5000/predict` with JSON such as:

```json
{
  "study_hours": 3.5,
  "attendance_percentage": 85
}
```

Example using `curl.exe` in Windows PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":3.5,\"attendance_percentage\":85}"
```

The API checks that attendance is between 0 and 100, and returns both inputs plus the prediction. **Important:** this version's model uses study hours only. Attendance is validated and returned, but does not affect the prediction until the model is retrained using attendance as a feature.

## Exercise 3 — Streamlit interface

Run:

```powershell
streamlit run streamlit_app.py
```

The browser interface asks for the student's name and study hours, provides a prediction button and result message, explains how the saved model generated the prediction, and displays a bonus chart.

## Notes

- The training records are illustrative sample data for completing the exercise. Replace them with the dataset supplied by your instructor if available.
- The model's output is for practice only and should not be treated as a reliable assessment of a real student's performance.
