# Setup
pip install scikit-learn pandas joblib flask streamlit

# Run order
python train_model.py              # Exercise 1 -> student_pass_model.joblib
python train_model_attendance.py   # Exercise 2 -> student_pass_model_attendance.joblib
python app.py                      # Exercise 2 (Flask API)
streamlit run streamlit_app.py     # Exercise 3

# Exercise 1 - Why no retraining is needed
Training is where the model learns patterns (its coefficients) from the data.
joblib.dump() saves those learned values into a file. joblib.load() restores the
exact same trained model, so it can predict immediately on new study hours.
Predicting only applies the already-learned values to new input, so training
again would only waste time and give the same result.

# Exercise 2 - Test with curl.exe (Windows)
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 85}"
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 150}"
# (Postman: POST http://127.0.0.1:5000/predict, Body > raw > JSON)
