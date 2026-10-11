import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

students = [
    {"study_hours": 1.0, "attendance_percentage": 80},
    {"study_hours": 5.0, "attendance_percentage": 80},
    {"study_hours": 9.0, "attendance_percentage": 80},
]

for s in students:
    row = pd.DataFrame([s])
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])
    print(f"Study hours: {s['study_hours']} -> P(pass)={prob:.2f} -> "
          f"{'PASS' if pred == 1 else 'FAIL'}")

print("""
Bakit hindi na kailangang mag-train ulit:
Ang training ang natututo ng model (coefficients at intercept).
Sine-save ito ng joblib.dump() at ibinabalik ng joblib.load()
nang eksakto ang parehong natutunan. Kaya prediction na lang
ang ginagawa, hindi na training.
""")