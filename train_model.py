import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# ─── 1. Bagong training data (may attendance na) ───
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'attendance':  [40, 50, 55, 65, 70, 85, 90, 95],
    'pass':        [0, 0, 0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# ─── 2. IMPORTANTE: Kunin ang DALAWANG features ───
X = df[['study_hours', 'attendance']]   # 🆕 2 columns na!
y = df['pass']

# ─── 3. Train ───
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
print("✅ Model retrained with 2 features!")

# ─── 4. Save ulit (override yung old model) ───
joblib.dump(model, 'student_pass_model.joblib')
print("✅ New model saved!")

# ─── 5. Test — dapat tumanggap ng 2 values na ───
test_cases = [
    [2, 40],   # mababa hours, mababa attendance → Fail?
    [5, 85],   # katamtamang hours, mataas attendance → Pass?
    [8, 50],   # mataas hours, mababa attendance → ?
]
for hours, att in test_cases:
    prediction = model.predict([[hours, att]])[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Hours: {hours}, Attendance: {att}% → {result}")