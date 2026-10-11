import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

study = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 3
result = ["Fail"] * 3 + ["Pass"] * 27

df = pd.DataFrame({"StudyHours": study[:30], "Result": result[:30]})

print("STUDENT DATASET")
print(df)

X = df[["StudyHours"]]
y = df["Result"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

dump(model, "student_pass_model.joblib")
print("\nModel saved as student_pass_model.joblib")

loaded_model = load("student_pass_model.joblib")
print("Model loaded from student_pass_model.joblib")

test_hours = [3, 7, 11]
predictions = loaded_model.predict(pd.DataFrame({"StudyHours": test_hours}))

print("\nPredictions using the loaded model:")
for hours, pred in zip(test_hours, predictions):
    print(f"Study Hours = {hours} -> Prediction: {pred}")

print("\nWhy the model does not need to be trained again:")
print("joblib.dump() serializes the trained model, including its learned")
print("parameters (for KNN, the stored training data and the k value).")
print("joblib.load() restores that exact model object, so it can make")
print("predictions immediately. Training is only needed again if the")
print("training data or the model type changes.")
