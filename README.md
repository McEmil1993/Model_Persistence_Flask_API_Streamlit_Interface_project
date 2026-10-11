Student Prediction System

Project Description

The Student Prediction System is a Python-based project that predicts whether a student will pass or fail based on study hours using a trained Decision Tree model.

Objectives

- Train a machine learning model to predict student results.
- Create an API that accepts student data and returns a prediction.
- Develop a simple web application for users to enter student information and view predictions.

Technologies Used

- Python
- Google Colab
- Scikit-learn
- Joblib
- Flask
- Streamlit
- GitHub

Project Exercises

Exercise 1: Machine Learning Model

Trains a Decision Tree classifier using sample study-hour data and saves the trained model as "student_pass_model.joblib".

Exercise 2: Flask API

Creates an API endpoint that accepts study hours and attendance percentage and returns a Pass/Fail prediction. The current "flask_api.py" uses a temporary rule-based prediction.

Exercise 3: Streamlit Web Application

Provides a simple web interface where users can enter a student's name and study hours to receive a prediction from the trained model.

Project Files

- "Student_Prediction_System.ipynb" — Notebook containing the project exercises.
- "flask_api.py" — Flask API implementation.
- "app.py" — Streamlit web application.
- "student_pass_model.joblib" — Saved machine learning model.
- "README.md" — Project documentation.

Note

The model was trained using sample data for demonstration and educational purposes. Its predictions should not be treated as an actual assessment of a student's academic performance.
