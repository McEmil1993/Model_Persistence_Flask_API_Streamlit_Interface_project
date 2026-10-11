# SkillTrack ML

SkillTrack ML is a classroom demonstration for IT Elective 3 — Machine Learning. It shows how to save and reuse a trained model, expose predictions through Flask, and present them in Streamlit.

The model uses a small synthetic dataset with study hours and attendance. Its categories are `Needs Support`, `Developing`, and `Ready to Advance`. These labels are illustrative only and should not be treated as a real academic assessment.

## Requirements

- Python 3.10 or newer
- Windows PowerShell
- Internet connection for installing packages

## Setup on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Exercise 1 — Model persistence

Train, save, reload, and test the model:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The script writes `skilltrack_model.joblib`, loads the saved file, and makes three example predictions. Model persistence means the fitted model can be loaded and reused without fitting it again every time the application starts. Retraining is needed when the model or training data must be updated.

## Exercise 2 — Flask API

Start the API in one terminal:

```powershell
.\.venv\Scripts\python.exe app.py
```

Keep that terminal open. In a second PowerShell terminal, test valid input:

```powershell
$body = @{
    study_hours = 5
    attendance_percentage = 90
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 3
```

Test invalid attendance:

```powershell
$body = @{
    study_hours = 5
    attendance_percentage = 120
} | ConvertTo-Json

try {
    Invoke-RestMethod `
        -Uri "http://127.0.0.1:5000/predict" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body
}
catch {
    $_.ErrorDetails.Message
}
```

The second request should return an error because attendance must be from 0 to 100. The API also validates study hours, required fields, numeric values, and finite numbers.

Open `http://127.0.0.1:5000/` in a browser to see the API description. A browser GET does not submit a prediction; use the POST examples above.

## Exercise 3 — Streamlit interface

Open another terminal in the project folder and run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Enter a learner name, set study hours and attendance, then select **Review learning progress**. The page displays a category, explanation, and comparison chart.

## Git submission

Use the repository URL and branch assigned by your instructor. Do not put passwords or personal access tokens in the remote URL or source files.

```powershell
git init
git add .
git config --local user.name "Your Name"
git config --local user.email "you@example.com"
git commit -m "Your Name - SkillTrack ML"
git branch -M Your_Branch
git remote add origin YOUR_TEACHER_REPOSITORY_URL
git push -u origin Your_Branch
```

If the repository already has a remote, check it with `git remote -v` before adding another. Replace `Your_Branch` and the placeholder values with the details assigned to you. Do not push to another student's branch.

## What the project demonstrates

1. **Model persistence:** `joblib.dump()` saves the fitted estimator and `joblib.load()` restores it.
2. **API deployment:** Flask validates a JSON request and returns a JSON prediction.
3. **Interface:** Streamlit collects inputs and displays the predicted category with a chart.
