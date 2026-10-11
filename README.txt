STUDENT PASS PREDICTION EXERCISES
=================================

IMPORTANT
This is a complete starter project because the exercise HTML did not include
the instructor's original Python files or dataset. The training data below is
small sample data for demonstrating the required workflow, not a real student
dataset.

REQUIREMENTS
- Python installed
- Internet connection for the first package installation

STEP 1: Open a terminal in this folder
In VS Code, open this folder, then choose Terminal > New Terminal.

STEP 2: (Recommended) Create a virtual environment
Windows:
    py -m venv .venv
    .venv\Scripts\activate

If activation is blocked in PowerShell, you can use Command Prompt instead,
or run this in the current PowerShell window:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

STEP 3: Install packages
    python -m pip install -r requirements.txt

STEP 4: Exercise 1 - train, save, load, and predict
    python train_model.py

This creates student_pass_model.joblib and prints three predictions.

STEP 5: Exercise 2 - start the Flask API
    python app.py

Keep that terminal open. Open a SECOND terminal and test with curl.exe:
    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 7, \"attendance_percentage\": 90}"

Try invalid attendance to check validation:
    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 7, \"attendance_percentage\": 120}"

The current model uses study_hours only. Attendance is accepted, validated, and
returned in JSON, but does not affect the prediction. To make attendance affect
the prediction, the training data and model must be updated to use both features,
then retrained.

STEP 6: Exercise 3 - start the Streamlit interface
Stop Flask with Ctrl+C. Then run:
    streamlit run streamlit_app.py

A browser tab should open. Enter a student name, choose study hours, and click
Predict Result.

ABOUT THE SAMPLE MODEL
The example uses a DecisionTreeClassifier trained on a small illustrative set:
1-5 study hours are labeled Fail and 6-10 study hours are labeled Pass.
This is only to demonstrate saving/loading and connecting a model to an API/UI.
It is not a reliable real-world predictor.
