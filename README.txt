STUDENT PREDICTION PROJECT
Based on the lesson: Model Persistence and Basic Deployment Using Flask API or Streamlit

FILES
- train_model.py: trains a small example model and saves student_pass_model.joblib
- app.py: Flask API with a /predict endpoint
- streamlit_app.py: simple Streamlit user interface and bonus chart
- requirements.txt: required Python packages

SETUP (VS CODE / WINDOWS)
1. Extract this ZIP file.
2. Open the extracted student_prediction_project folder in VS Code.
3. Open Terminal > New Terminal.
4. Optional but recommended: create a virtual environment:
   py -m venv .venv
   .venv\Scripts\activate
5. Install the packages:
   python -m pip install -r requirements.txt
6. Train and save the model:
   python train_model.py

RUN THE STREAMLIT INTERFACE
   streamlit run streamlit_app.py
Open the local URL printed in the terminal (usually http://localhost:8501).

RUN THE FLASK API
Open another terminal in the same folder and run:
   python app.py
The API usually runs at http://127.0.0.1:5000.

TEST THE API IN POWERSHELL
   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":90}"

You can also send a POST request to http://127.0.0.1:5000/predict in Postman.
Choose Body > raw > JSON and enter:
{
  "study_hours": 5,
  "attendance_percentage": 90
}

IMPORTANT NOTE
This is a small teaching example with made-up sample data, not a reliable real-world student assessment model.
The model uses study_hours only. The API validates and returns attendance_percentage because the exercise asks for it, but attendance does not affect the prediction unless you change the training data and train the model with both features.
