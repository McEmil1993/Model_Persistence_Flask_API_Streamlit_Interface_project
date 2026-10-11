STUDENT PREDICTION PROJECT — MOBILE GUIDE

FILES
- train_model.py: trains and saves the example model; prints three predictions.
- student_pass_model.joblib: generated when train_model.py runs.
- app.py: Flask JSON API.
- streamlit_app.py: Streamlit interface.
- requirements.txt: Python libraries.

EASIEST MOBILE OPTION
1. Open https://colab.research.google.com/ in your phone browser.
2. Create a new notebook.
3. Install packages in a cell:
   !pip install scikit-learn joblib flask streamlit pandas
4. Copy the code from train_model.py into a notebook cell and run it.
5. Download student_pass_model.joblib from the Colab Files panel if you want to move it elsewhere.
6. To run Flask or Streamlit continuously, use a Python hosting service that supports web apps, or a suitable mobile Linux/Python environment. Colab sessions are temporary and are not ideal for hosting public APIs.

RUN LOCALLY (if you have a Python environment)
pip install -r requirements.txt
python train_model.py
python app.py
streamlit run streamlit_app.py

TEST FLASK
POST to http://127.0.0.1:5000/predict with JSON:
{
  "study_hours": 5,
  "attendance_percentage": 90
}

Expected response shape:
{
  "study_hours": 5.0,
  "attendance_percentage": 90.0,
  "prediction_class": 1,
  "prediction": "Likely to Pass",
  "note": "Attendance is validated and returned, but does not affect this model's prediction."
}

MODEL NOTE
The example uses only study hours as the model feature. Attendance is accepted, checked, and returned by Flask, but it does not influence the prediction. To make attendance influence predictions, train a new model using two input features: study hours and attendance percentage, with suitable real training data and labels.

PERSISTENCE EXPLANATION
joblib.dump() saves the trained model to a file. joblib.load() restores that already-trained model. Predictions use the saved model's learned decision rules, so training is not repeated for each prediction.
