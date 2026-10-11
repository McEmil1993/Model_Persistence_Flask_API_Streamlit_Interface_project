import unittest

import joblib

import app as api
from train_model import MODEL_PATH, train_and_save_model


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_and_save_model()
        api.model = joblib.load(MODEL_PATH)

    def setUp(self):
        self.client = api.app.test_client()

    def test_returns_both_inputs_and_study_hours_prediction(self):
        response = self.client.post(
            "/predict",
            json={"study_hours": 8, "attendance_percentage": 92},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "study_hours": 8.0,
                "attendance_percentage": 92.0,
                "prediction": 1,
                "prediction_label": "Pass",
            },
        )

    def test_rejects_attendance_outside_zero_to_one_hundred(self):
        for attendance in (-1, 101):
            with self.subTest(attendance=attendance):
                response = self.client.post(
                    "/predict",
                    json={"study_hours": 8, "attendance_percentage": attendance},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("attendance_percentage", response.get_json()["error"])

    def test_rejects_missing_or_non_json_input(self):
        missing_field = self.client.post("/predict", json={"study_hours": 8})
        invalid_body = self.client.post("/predict", data="not json")

        self.assertEqual(missing_field.status_code, 400)
        self.assertEqual(invalid_body.status_code, 400)


if __name__ == "__main__":
    unittest.main()