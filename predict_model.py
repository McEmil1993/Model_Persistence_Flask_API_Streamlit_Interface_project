
import joblib

model = joblib.load("student_pass_model.joblib")

hours = [[2], [5], [8]]

for hour in hours:
    result = model.predict(hour)
    prediction = "Pass" if result[0] == 1 else "Fail"
    print("Study hours:", hour[0])
    print("Prediction:", prediction)
