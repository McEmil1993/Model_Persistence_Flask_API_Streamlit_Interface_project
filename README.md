# Student Performance Prediction

**Student:** Rica Alba Prequencia

A learning activity with three parts:
1. Model persistence using `joblib.dump()` and `joblib.load()`.
2. Flask API accepting study hours and attendance percentage.
3. Streamlit dashboard with a professional black-and-blue UI.

> This is a classroom demonstration using illustrative training data. The model uses **study hours only**. Attendance is validated and returned by the API, but it does not affect the prediction.

## 1. Open in VS Code

Extract the project ZIP, open the `rica_student_prediction` folder in VS Code, then open Terminal.

## 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## 3. Install dependencies and train the model

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

This creates `models/student_pass_model.joblib`. You only need to retrain when the training data/model changes.

## 4. Run the Streamlit interface

```powershell
python -m streamlit run streamlit_app.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`.

## 5. Run the Flask API

Open a second terminal in the same folder and activate the virtual environment, then run:

```powershell
python app.py
```

The API runs at `http://127.0.0.1:5000`.

## 6. Test the API with curl.exe

Keep Flask running, open another terminal in the project folder, then run:

```powershell
curl.exe -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" --data-binary "@request.json"
```

Expected JSON fields include `study_hours`, `attendance_percentage`, `prediction`, `result`, `pass_probability`, and a note explaining the model feature. The probability can vary with model/library versions.

Try invalid attendance, such as `{"study_hours": 6, "attendance_percentage": 120}`, to verify validation.

## 7. Git commands from the activity

Configure your own Git identity first, then initialize and commit:

```powershell
git init
git add .
git config user.name "Rica Alba Prequencia"
git config user.email "YOUR_EMAIL"
git commit -m "Rica Alba Prequencia"
git branch -M main
```

Add your repository URL with `git remote add origin YOUR_REPOSITORY_URL`, then push with `git push -u origin main`.

**Security:** Never put a GitHub personal access token in a remote URL, a source file, or a shared screenshot. If a token has been exposed, revoke it and create a new one.
