# 🎓 Student Performance Predictor

A beginner-friendly Machine Learning project developed using Python, Scikit-learn, Joblib, Flask, and Streamlit. This application predicts student performance based on **study hours and attendance percentage**.

This project was created for **IT Elective 3 — AI: Machine Learning** to demonstrate model persistence, prediction, API development, and basic web application deployment.

> **Disclaimer:** This project uses a small, fictional dataset for educational purposes. Its predictions are not scientifically validated and should not be used to make actual academic decisions.

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Train and Save the Model](#-train-and-save-the-model)
- [Run the Streamlit Application](#-run-the-streamlit-application)
- [Run the Flask API](#-run-the-flask-api)
- [Test the Flask API](#-test-the-flask-api)
- [How Model Persistence Works](#-how-model-persistence-works)
- [Troubleshooting](#-troubleshooting)
- [Limitations](#-limitations)
- [Learning Outcomes](#-learning-outcomes)

## 📖 Project Overview

The Student Performance Predictor demonstrates how a trained Machine Learning model can be saved and integrated into a web application.

Users enter a student's name, study hours, and attendance percentage. The application passes the numerical features to a trained Decision Tree Classifier, which predicts either **Pass** or **Fail**.

The project supports two deployment approaches:

- **Streamlit:** Provides an interactive web interface for users.
- **Flask API:** Provides a backend endpoint that accepts JSON requests and returns predictions as JSON responses.

Both applications can reuse the same saved Machine Learning model.

## ✨ Features

### 1. Machine Learning Prediction

- Uses a Decision Tree Classifier.
- Accepts study hours as an input feature.
- Accepts attendance percentage as an input feature.
- Produces a Pass or Fail prediction.

### 2. Model Persistence

- Saves the trained model using Joblib.
- Loads the saved model without retraining it for every prediction.
- Stores the model in the `models/` directory.

### 3. Streamlit Web Interface

- Student name input.
- Study-hours input.
- Attendance percentage input.
- Predict Performance button.
- Prediction result display.
- Explanation of the prediction.
- Chart comparing attendance percentages with predicted classes while holding study hours constant.

### 4. Flask API

- Accepts JSON requests.
- Validates study hours and attendance percentage.
- Returns prediction results in JSON format.
- Provides an endpoint for integration with other applications.

### 5. Input Validation

- Study hours must be between 0 and 24.
- Attendance must be between 0% and 100%.
- Student name is required in the Streamlit interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Scikit-learn | Machine Learning model |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web interface |
| Flask | Backend API |
| Pandas | Data preparation for visualization |
| Git | Version control |
| GitHub | Source code hosting |

## 📁 Project Structure

```text
ml_model_deployment/
│
├── train_model.py
├── test_model.py
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── student_pass_model.joblib
│
└── .venv/
```

### File Descriptions

| File or Folder | Description |
|---|---|
| `train_model.py` | Trains the model using study hours and attendance percentage. |
| `test_model.py` | Loads the saved model and tests predictions. |
| `app.py` | Runs the Flask prediction API. |
| `streamlit_app.py` | Runs the Streamlit web interface. |
| `requirements.txt` | Lists the Python packages required by the project. |
| `README.md` | Contains project documentation. |
| `.gitignore` | Excludes files that should not be committed to Git. |
| `models/` | Stores the saved Machine Learning model. |
| `.venv/` | Contains the local Python virtual environment. |

The `.venv/` directory should not be uploaded to GitHub. The saved model may also be excluded from GitHub and generated locally by running the training script.

## 📋 Prerequisites

Before running the project, install:

- Python 3.10 or later
- Visual Studio Code or another Python editor
- Git
- A modern web browser

Basic knowledge of Python and command-line operations is helpful.

## ⚙️ Installation

### Step 1: Clone the GitHub repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-predictor.git
```

Navigate into the project folder:

```bash
cd student-performance-predictor
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Alternatively, use Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet, create it with the following contents:

```text
scikit-learn
joblib
streamlit
flask
pandas
```

Then run the installation command again.

## 🧠 Train and Save the Model

The model must be trained before running the Streamlit application or Flask API for the first time.

Execute:

```bash
python train_model.py
```

The training script should:

1. Prepare the sample training data.
2. Define study hours and attendance percentage as the two input features.
3. Train a Decision Tree Classifier.
4. Create the `models/` directory if necessary.
5. Save the model as `models/student_pass_model.joblib`.

Example output:

```text
Model trained successfully!
Features: Study hours and attendance percentage.
Saved to models/student_pass_model.joblib
```

The exact output depends on the messages in your training script.

**Important:** The model must be trained using both features in this order:

1. Study hours
2. Attendance percentage

The prediction code must use the same feature order.

## 🚀 Run the Streamlit Application

After training the model, start the web interface:

```bash
streamlit run streamlit_app.py
```

Open the address displayed in your terminal. The default is usually:

http://localhost:8501

### How to Use Streamlit

1. Enter the student's name.
2. Enter study hours per day.
3. Enter attendance percentage.
4. Click **Predict Performance**.
5. View the predicted Pass or Fail result.
6. Review the attendance chart.

The chart shows predicted classes at different attendance percentages while keeping the entered study hours constant.

## 🌐 Run the Flask API

The Flask API allows another application to submit student information and receive a prediction.

Start the API:

```bash
python app.py
```

The API will normally be available at:

http://127.0.0.1:5000

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns basic API information. |
| POST | `/predict` | Accepts student data and returns a prediction. |

## 🧪 Test the Flask API

Send a POST request to:

```text
http://127.0.0.1:5000/predict
```

Set the request's `Content-Type` header to `application/json`.

### Sample Request

```json
{
  "study_hours": 5,
  "attendance_percentage": 80
}
```

### Sample Response

```json
{
  "study_hours": 5.0,
  "attendance_percentage": 80.0,
  "prediction": "Pass",
  "note": "Prediction is based on study hours and attendance."
}
```

The exact response message depends on the implementation in `app.py`.

You can test the endpoint using Postman or Windows `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":5,\"attendance_percentage\":80}"
```

### Input Validation Tests

| Test Case | Expected Behavior |
|---|---|
| Valid study hours and attendance | Return a prediction. |
| Attendance above 100 | Return a validation error. |
| Negative attendance | Return a validation error. |
| Missing required field | Return a validation error. |
| Non-numeric input | Return a validation error. |

The Flask API and Streamlit interface must both load a model trained with the same two features.

## 💾 How Model Persistence Works

Model persistence is the process of saving a trained Machine Learning model so it can be reused later without repeating training.

### Saving the Model

The training script uses Joblib:

```python
joblib.dump(
    model,
    "models/student_pass_model.joblib"
)
```

### Loading the Model

The application loads the saved model:

```python
model = joblib.load(
    "models/student_pass_model.joblib"
)
```

### Making a Prediction

The application provides both features:

```python
prediction = model.predict([
    [study_hours, attendance]
])[0]
```

The output is converted into a readable result:

```python
result = "Pass" if prediction == 1 else "Fail"
```

Once saved, the model can be loaded by different applications without being retrained for every prediction.

Retraining is required when the model needs to learn from updated data or when its training process or features change.

## 🐛 Troubleshooting

### 1. Model file not found

**Error:**

```text
FileNotFoundError:
models/student_pass_model.joblib
```

**Solution:**

Run the training script first:

```bash
python train_model.py
```

### 2. No module named `streamlit`

**Solution:**

```bash
python -m pip install streamlit
```

### 3. Number of features does not match

This can occur when the saved model was trained with only study hours.

**Solution:**

Retrain the model using both study hours and attendance percentage:

```bash
python train_model.py
```

Restart the Streamlit application or Flask API afterward.

### 4. Port already in use

For Streamlit:

```bash
streamlit run streamlit_app.py --server.port 8502
```

For Flask, change the port in `app.py` and use the same port when testing requests.

### 5. GitHub clone does not contain the model file

If the `.joblib` file is excluded by `.gitignore`, this is expected.

Run the training script after cloning:

```bash
python train_model.py
```

This generates the model locally.

## ⚠️ Limitations

- The training data is fictional and very small.
- Predictions are not validated against real student outcomes.
- The model is intended for learning, not actual academic assessment.
- The chart displays class labels rather than prediction probabilities.
- The application does not include authentication or persistent student records.
- The Flask development server is not intended for production deployment.

For real-world use, the model would require representative data, suitable evaluation metrics, privacy protections, and careful validation.

## 🎯 Learning Outcomes

After completing this project, students should be able to:

1. Explain model persistence and deployment.
2. Train a Machine Learning classification model.
3. Save and load models using Joblib.
4. Make predictions using multiple input features.
5. Build an interactive Streamlit application.
6. Develop and test a Flask API.
7. Validate user input.
8. Visualize model predictions.
9. Explain the limitations of a Machine Learning model.
10. Use Git and GitHub to manage and share source code.

## 👨‍💻 Academic Information

**Project Name:** Student Performance Predictor  
**Subject:** IT Elective 3 — AI: Machine Learning  
**Programming Language:** Python  
**Model:** Decision Tree Classifier  
**Deployment:** Streamlit and Flask  
**Purpose:** Educational demonstration of model persistence and basic deployment

---

*Developed for educational purposes. Predictions are demonstrations and should not be used as the sole basis for academic decisions.*