# LearnLens — IT Elective 3 Machine Learning Activity

LearnLens is a small classroom project with three parts: saving and reusing a trained model, serving predictions through Flask, and testing the model through Streamlit.

The model uses two inputs:
- `study_hours`: study time per day, from 0 to 24
- `attendance_percentage`: attendance, from 0 to 100

It predicts one of three demonstration categories: `Needs Support`, `On Track`, or `Strong Progress`. The training data is synthetic and is not suitable for evaluating real students.

## 1. Windows setup

Open PowerShell in this folder. If you use Git Bash, use forward-slash paths or switch to PowerShell for these commands.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Exercise 1 — Model persistence

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The script trains a model, saves it as `learnlens_model.joblib`, loads the saved file, and makes three sample predictions. The model does not need to be trained again just to reuse it; run the training script again only when you want to rebuild or update the model.

## 3. Exercise 2 — Flask API

Start the API in one terminal and leave it running:

```powershell
.\.venv\Scripts\python.exe app.py
```

In a second PowerShell terminal, send a valid request:

```powershell
$body = @{
    study_hours = 5
    attendance_percentage = 90
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 3
```

Test attendance validation with an invalid value:

```powershell
$body = @{
    study_hours = 5
    attendance_percentage = 120
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body $body
```

The invalid request should return HTTP 400 with an error message. PowerShell may display an error for this expected rejection.

## 4. Exercise 3 — Streamlit interface

In another terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Enter a learner name, set study hours and attendance, then select **Review learning progress**. The page shows a category, explanation, and chart.

## 5. Suggested screenshots

1. Exercise 1 terminal showing the saved model and three predictions.
2. Flask valid JSON response.
3. Flask invalid attendance request being rejected.
4. LearnLens Streamlit page showing a prediction and chart.

## 6. Git submission

First confirm the instructor's repository URL and required branch name. Do not put a password or personal access token in the remote URL.

```bash
git init
git status --short
git add .gitignore README.md app.py requirements.txt streamlit_app.py train_model.py learnlens_model.joblib
git config --local user.name "YOUR NAME"
git config --local user.email "YOUR EMAIL"
git commit -m "YOUR NAME - LearnLens ML Activity"
git branch -M YOUR_BRANCH
git remote add origin REPOSITORY_URL
git push -u origin YOUR_BRANCH
```

Replace the uppercase placeholders with your own details. If Git says a remote named `origin` already exists, check it with `git remote -v` before changing anything. Do not push `.venv/`.
