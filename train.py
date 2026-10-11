import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

rng = np.random.default_rng(42)

rows = []
labels = []

# 200 PASS examples: hours 5-10, attendance 75-100
for _ in range(200):
    h = rng.uniform(5.0, 10.0)
    a = rng.uniform(75.0, 100.0)
    rows.append([h, a])
    labels.append(1)

# 100 FAIL examples: low hours (1 to 4.9), any attendance
for _ in range(100):
    h = rng.uniform(1.0, 4.9)
    a = rng.uniform(40.0, 100.0)
    rows.append([h, a])
    labels.append(0)

# 100 FAIL examples: good hours (5-10), low attendance (40 to 74.9)
for _ in range(100):
    h = rng.uniform(5.0, 10.0)
    a = rng.uniform(40.0, 74.9)
    rows.append([h, a])
    labels.append(0)

X = np.array(rows, dtype=float)
y = np.array(labels, dtype=int)

print("Dataset shape:", X.shape)
print("Pass count:", int(y.sum()))
print("Fail count:", int((y == 0).sum()))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print()
print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
print(f"Test  accuracy: {accuracy_score(y_test,  model.predict(X_test)):.4f}")
print()
print(classification_report(y_test, model.predict(X_test), target_names=["Fail", "Pass"]))

# Feature order: [study_hours, attendance_percentage]
joblib.dump(model, "student_pass_model.joblib")
print("Saved: student_pass_model.joblib")

# Sanity check
print()
print("Sanity checks:")
for h, a in [(5, 85), (2, 60), (8, 90), (6, 50)]:
    p = model.predict_proba([[h, a]])[0]
    label = "Pass" if p[1] >= 0.5 else "Fail"
    print(f"  hours={h:>4}  attendance={a:>3}  ->  {label}  (P(Pass)={p[1]:.3f})")