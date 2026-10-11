import os
import unittest
from pathlib import Path


class AttendanceModelFallbackTest(unittest.TestCase):
    def test_missing_attendance_model_is_created(self):
        model_path = Path(__file__).resolve().parent / "student_pass_model_attendance.joblib"
        if model_path.exists():
            model_path.unlink()

        import app

        self.assertTrue(model_path.exists())
        prediction = app.model.predict([[2, 80]])[0]
        self.assertIn(int(prediction), (0, 1))


if __name__ == "__main__":
    unittest.main()
