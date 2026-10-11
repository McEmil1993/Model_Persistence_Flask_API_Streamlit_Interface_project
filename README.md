# Student Prediction Project

A Python-based student performance prediction system using Machine Learning, Flask API, and Streamlit.

## Project Features

1. **Model Persistence** – Trains a Logistic Regression model and saves it using `joblib`.
2. **Flask API** – Accepts student study hours and attendance percentage, validates inputs, and returns a prediction in JSON format.
3. **Streamlit Web App** – Provides a user interface where users can enter student information and view prediction results.

## Project Structure

```text
student_prediction_project/
│
├── train_model.py
├── app.py
├── streamlit_app.py
├── requirements.txt
├── student_pass_model.joblib
└── README.md
```

## Requirements

- Python
- scikit-learn
- joblib
- Flask
- Streamlit
- Requests
- Pandas
- Matplotlib
- NumPy

## Installation

### Step 1: Open the Project Folder

Open Git Bash and run:

```bash
cd ~/OneDrive/Desktop/student_prediction_project
```

### Step 2: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Exercise 1: Model Persistence

Train the model and save it to a `.joblib` file.

```bash
python train_model.py
```

The program will:

- Train the Logistic Regression model.
- Save the trained model as `student_pass_model.joblib`.
- Load the saved model using `joblib.load()`.
- Make three sample student predictions.
- Explain why the model does not need to be retrained every time.

## Exercise 2: Flask API

### Step 1: Start the Flask Server

In Git Bash, run:

```bash
python app.py
```

Keep this terminal open. The API should run at:

`http://127.0.0.1:5000`

### Step 2: Test the Prediction API

Open a second Git Bash terminal and run:

```bash
cd ~/OneDrive/Desktop/student_prediction_project
```

Send a prediction request:

```bash
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":80}"
```

Example response:

```json
{
  "attendance_percentage": 80.0,
  "pass_probability": 0.9974,
  "prediction": "Pass",
  "status": "success",
  "study_hours": 5.0
}
```

The exact prediction and probability depend on the trained model.

### Step 3: Test Attendance Validation

Send an invalid attendance value:

```bash
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":150}"
```

Expected response:

```json
{
  "error": "attendance_percentage must be between 0 and 100"
}
```

The API validates attendance to ensure it stays between 0 and 100.

## Exercise 3: Streamlit Web Application

### Step 1: Open a Third Git Bash Terminal

Run:

```bash
cd ~/OneDrive/Desktop/student_prediction_project
```

### Step 2: Start Streamlit

```bash
python -m streamlit run streamlit_app.py
```

### Step 3: Open the Application

Open this URL in your browser:

`http://localhost:8501`

### Step 4: Use the Application

1. Enter the student's name.
2. Enter the number of study hours.
3. Select the attendance percentage.
4. Click **Predict Result**.
5. View the prediction and pass probability.
6. Review the explanation and bonus chart.

**Important:** Keep the Flask server running while using Streamlit because the web application sends prediction requests to the Flask API.

## Commands Summary

| Task | Command |
|---|---|
| Open project folder | `cd ~/OneDrive/Desktop/student_prediction_project` |
| Install dependencies | `python -m pip install -r requirements.txt` |
| Train and save model | `python train_model.py` |
| Run Flask API | `python app.py` |
| Run Streamlit app | `python -m streamlit run streamlit_app.py` |
| Test API | `curl.exe -X POST ...` |

## Why Model Persistence Is Important

Model persistence allows a trained machine learning model to be saved and reused without training it again every time the application starts. The `joblib` library saves the model in a file that can be loaded later for predictions.

## Conclusion

This project demonstrates how machine learning can be integrated into a web application. It combines model persistence, a Flask API, input validation, and a Streamlit interface to predict student performance based on study hours and attendance percentage.

**Note:** This is a learning project using a small sample dataset. Its predictions are for demonstration purposes and should not be treated as reliable academic assessments.
