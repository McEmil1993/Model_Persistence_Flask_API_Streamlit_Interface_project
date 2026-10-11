from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier

# Small example dataset: the model learns from study hours only.
# Replace these sample values with your teacher's dataset if one was provided.
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

model_path = Path(__file__).resolve().parent / "student_pass_model.joblib"
joblib.dump(model, model_path)

print(f"Model trained and saved to: {model_path.name}")
print("The model uses study hours only.")
