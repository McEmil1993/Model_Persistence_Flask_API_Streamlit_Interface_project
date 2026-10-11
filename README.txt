STUDENT PASS PREDICTION EXERCISES — TWO INPUT FEATURES
======================================================

IMPORTANT NOTE ABOUT THE DATA
The instructor's HTML did not include the original Python files or a training
 dataset. This project therefore uses illustrative sample training data. It
is not trained on actual student records and is not a reliable real-world
academic predictor. The sample labels use an example pattern (at least 6 study
hours and at least 75% attendance = Pass) to demonstrate how a model can use
two input features. This is only a teaching example.

REQUIREMENTS
- Python installed
- Internet connection for package installation

STEP 1: Open a terminal in this folder.

STEP 2: (Recommended) Create and activate a virtual environment.
    py -m venv .venv
    .venv\Scripts\activate

If PowerShell blocks activation, run this for the current terminal only:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

STEP 3: Install packages.
    python -m pip install -r requirements.txt

STEP 4: Train and save the UPDATED two-input model.
    python train_model.py

IMPORTANT: Run this after replacing/updating these files. It overwrites
student_pass_model.joblib with the new model trained using both inputs.

STEP 5: Exercise 2 — run Flask API.
    python app.py

Keep that terminal open. In a SECOND terminal, run:
    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{\"study_hours\": 7, \"attendance_percentage\": 90}'

Try a different attendance value while keeping study hours the same:
    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{\"study_hours\": 7, \"attendance_percentage\": 60}'

Test validation with invalid attendance:
    curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{\"study_hours\": 7, \"attendance_percentage\": 120}'

STEP 6: Exercise 3 — run Streamlit.
Stop Flask with Ctrl+C, then run:
    streamlit run streamlit_app.py

The UI now accepts student name, study hours, and attendance percentage. It
uses both numeric inputs to generate a prediction. The bonus chart compares
study hours with the predicted class while holding attendance constant. An
additional chart compares attendance with the predicted class while holding
study hours constant.

MODEL PERSISTENCE
The model is saved as student_pass_model.joblib and loaded by both apps. It
must be retrained once after the model inputs change from one feature to two.
After that, the saved two-feature model can be loaded for future predictions
without retraining every time.
