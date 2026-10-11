# John Mark T. Cajes

# Student Pass/Fail Machine Learning Activities

This project demonstrates how to train and save a Machine Learning model, use
the saved model through a Flask API, and present it through a Streamlit web
interface.

## Project files

```text
Activity_final/
├── train_model.py
├── test_model.py
├── app.py
├── streamlit_app.py
├── student_pass_model.joblib
├── requirements.txt
└── README.md
```

## Initial setup

Open PowerShell or the VS Code terminal and go to the project folder:

```powershell
cd C:\Python\Activity_final
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

---

## Activity 1 — Model Persistence

Activity 1 trains a pass/fail model, saves it as
`student_pass_model.joblib`, loads the saved model, and makes three
predictions.

### Test Activity 1

Train and save the model:

```powershell
python train_model.py
```

Expected output:

```text
Model trained and saved to: C:\Python\Activity_final\student_pass_model.joblib
```

The `student_pass_model.joblib` file should now exist in the project folder.

Load the saved model and make three predictions:

```powershell
python test_model.py
```

Expected output:

```text
Predictions from the saved model:
- 2 study hours: Fail
- 5 study hours: Pass
- 8 study hours: Pass

The model was loaded from student_pass_model.joblib, so it did not need to be trained again.
```

### Why the model does not need retraining

`joblib.dump()` saves the trained model and the parameters it learned from the
training data. `joblib.load()` restores those parameters, allowing the model
to make new predictions without calling `fit()` again.

---

## Activity 2 — Flask API

Activity 2 provides a `POST /predict` endpoint. It accepts `study_hours` and
`attendance_percentage`, validates the values, and returns a JSON prediction.

The current model was trained using only `study_hours`. Attendance is accepted,
validated, and returned by the API, but it does not affect the prediction.

### Start the Flask API

Run this command in the first terminal:

```powershell
python app.py
```

Expected terminal message:

```text
Running on http://127.0.0.1:5000
```

Keep this terminal running while testing the API. Press `Ctrl+C` when you want
to stop the server.

### Test with curl.exe

Open a second PowerShell terminal and create the JSON request body:

```powershell
$body = '{"study_hours":5,"attendance_percentage":90}'
```

Send the request:

```powershell
curl.exe -X POST "http://127.0.0.1:5000/predict" `
  -H "Content-Type: application/json" `
  --data $body
```

Expected response:

```json
{
  "attendance_percentage": 90.0,
  "prediction": "Pass",
  "study_hours": 5.0
}
```

#### Test invalid attendance with curl.exe

```powershell
$body = '{"study_hours":5,"attendance_percentage":120}'

curl.exe -X POST "http://127.0.0.1:5000/predict" `
  -H "Content-Type: application/json" `
  --data $body
```

Expected response:

```json
{
  "error": "attendance_percentage must be between 0 and 100."
}
```

The invalid request should return HTTP status `400 Bad Request`. This is the
expected result and confirms that validation is working.

### Test with Postman

Postman tests the Flask API in `app.py`. It does not test
`streamlit_app.py`.

1. Open a terminal in `C:\Python\Activity_final` and start the Flask API:

   ```powershell
   python app.py
   ```

2. Keep that terminal running. The API must remain available at
   `http://127.0.0.1:5000` while Postman sends requests.
3. Open Postman and create a new HTTP request.
4. Change the request method from `GET` to `POST`.
5. Enter this request URL:

   ```text
   http://127.0.0.1:5000/predict
   ```

6. Select **Body**.
7. Select **raw**.
8. Select **JSON** as the body format. Postman should add the
   `Content-Type: application/json` header automatically.
9. Enter this JSON body:

   ```json
   {
     "study_hours": 5,
     "attendance_percentage": 90
   }
   ```

10. Select **Send**.

Postman should show status `200 OK` and a response similar to:

```json
{
  "attendance_percentage": 90.0,
  "prediction": "Pass",
  "study_hours": 5.0
}
```

To test validation in Postman, change `attendance_percentage` to `120` and
select **Send** again. Postman should show status `400 Bad Request` and this
response:

```json
{
  "error": "attendance_percentage must be between 0 and 100."
}
```

Press `Ctrl+C` in the Flask terminal after testing is complete.

---

## Activity 3 — Streamlit Interface

Activity 3 provides a graphical interface with a student-name field, a
study-hours input, a prediction button, a result message, an explanation, and
a bonus chart.

### Test Activity 3

Streamlit loads the saved model directly and provides a browser interface. It
does not require the Flask API or Postman.

Open a terminal in the project folder:

```powershell
cd C:\Python\Activity_final
```

Start Streamlit:

```powershell
python -m streamlit run streamlit_app.py
```

Streamlit should open the application in a browser. If it does not open
automatically, visit:

```text
http://localhost:8501
```

Test the interface:

1. Enter a student name, such as `Alex`.
2. Enter `5` for the study hours.
3. Select **Generate prediction**.
4. Confirm that the result says `Alex is predicted to pass.`
5. Confirm that the prediction explanation appears.
6. Confirm that the chart displays the study-hours value and predicted class.

Also test the name validation:

1. Remove the student name.
2. Select **Generate prediction**.
3. Confirm that a toast notification says
   `Enter the student's name before generating a prediction.`

Press `Ctrl+C` in the terminal to stop Streamlit when testing is complete.

## Expected completion checklist

- [ ] Activity 1 saves `student_pass_model.joblib`.
- [ ] Activity 1 loads the saved model and produces three predictions.
- [ ] Activity 2 returns a prediction for valid JSON input.
- [ ] Activity 2 rejects attendance below `0` or above `100`.
- [ ] Activity 2 works with both `curl.exe` and Postman.
- [ ] Activity 3 accepts a student name and study hours.
- [ ] Activity 3 displays a result, explanation, and chart.
