import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 120

study_hours = np.round(rng.uniform(0, 10, n), 1)
attendance = np.round(rng.uniform(30, 100, n), 0)

# Higher score = more likely to pass
score = 0.55 * study_hours + 0.045 * attendance - 5.0 + rng.normal(0, 0.8, n)
passed = (score > 0).astype(int)

pd.DataFrame({
    "study_hours": study_hours,
    "attendance_percentage": attendance,
    "passed": passed,
}).to_csv("students.csv", index=False)

print(f"Created students.csv with {n} rows, {passed.mean():.0%} passed")
