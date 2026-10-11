# Exercise 1 - Load the saved model and make 3 predictions (NO retraining)
import joblib

model = joblib.load("student_pass_model.joblib")

for hours in [1.5, 4.5, 7]:
    pred = model.predict([[hours]])[0]
    print(f"{hours} study hours -> {'Pass' if pred == 1 else 'Fail'}")

# Why no retraining is needed:
# Training (fit) is where the model learns its parameters (coefficients and
# intercept). joblib.dump() saves those learned parameters to a file, and
# joblib.load() restores them. The loaded model already "knows" the pattern,
# so predict() just applies the saved formula to the new input.
