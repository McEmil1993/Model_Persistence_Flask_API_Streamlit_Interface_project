from sklearn.linear_model import LogisticRegression
from joblib import dump

X = [
    [1, 30],  # Fail
    [2, 60],  # Fail 
    [3, 70],  # Fail
    [4, 75],  # Pass
    [5, 85],  # Pass
    [7, 90]   # Pass
]
y = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]

model = LogisticRegression()
model.fit(X, y)

dump(model, 'student_pass_model.joblib')
print("Model retrained with stricter rules!")