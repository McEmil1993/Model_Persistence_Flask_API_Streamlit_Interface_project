"""
test_api.py

Automated tests for the Flask API (Exercise 2) and for model persistence (Exercise 1).
No server needs to be running. Run train_model.py first, then:

    python -m unittest -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import app as api  # loads student_pass_model_attendance.joblib


class PredictEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()

    def post(self, payload):
        return self.client.post("/predict", json=payload)

    def post_raw(self, body, content_type="application/json"):
        return self.client.post("/predict", data=body, content_type=content_type)

    # --- successful predictions ---------------------------------------------
    def test_valid_request_returns_both_values_and_prediction(self):
        response = self.post({"study_hours": 6, "attendance_percentage": 90})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["study_hours"], 6)
        self.assertEqual(body["attendance_percentage"], 90)
        self.assertIn(body["prediction"], ("Pass", "Fail"))
        self.assertTrue(0 <= body["pass_probability"] <= 1)

    def test_strong_student_passes_and_weak_student_fails(self):
        self.assertEqual(self.post({"study_hours": 9, "attendance_percentage": 95}).get_json()["prediction"], "Pass")
        self.assertEqual(self.post({"study_hours": 1, "attendance_percentage": 60}).get_json()["prediction"], "Fail")

    def test_attendance_influences_the_prediction(self):
        low = self.post({"study_hours": 5, "attendance_percentage": 55}).get_json()["pass_probability"]
        high = self.post({"study_hours": 5, "attendance_percentage": 100}).get_json()["pass_probability"]
        self.assertGreater(high, low)

    # --- attendance validation (0 to 100) -----------------------------------
    def test_attendance_boundaries_are_accepted(self):
        for value in (0, 100):
            self.assertEqual(self.post({"study_hours": 5, "attendance_percentage": value}).status_code, 200)

    def test_attendance_outside_range_is_rejected(self):
        for value in (-0.1, 100.1, 120, -5):
            response = self.post({"study_hours": 5, "attendance_percentage": value})
            self.assertEqual(response.status_code, 400, value)
            self.assertIn("between 0 and 100", response.get_json()["errors"][0])

    # --- other validation -----------------------------------------------------
    def test_negative_study_hours_rejected(self):
        self.assertEqual(self.post({"study_hours": -1, "attendance_percentage": 80}).status_code, 400)

    def test_missing_fields_rejected(self):
        self.assertEqual(self.post({"study_hours": 5}).status_code, 400)
        self.assertEqual(self.post({"attendance_percentage": 80}).status_code, 400)
        self.assertEqual(self.post({}).status_code, 400)

    def test_wrong_types_rejected(self):
        for bad in ("five", None, True, [5], {"a": 1}):
            response = self.post({"study_hours": bad, "attendance_percentage": 80})
            self.assertEqual(response.status_code, 400, repr(bad))

    def test_all_problems_are_reported_together(self):
        response = self.post({"study_hours": "x", "attendance_percentage": 500})
        self.assertEqual(len(response.get_json()["errors"]), 2)

    def test_nan_and_infinity_rejected(self):
        for literal in ("NaN", "Infinity"):
            response = self.post_raw('{"study_hours": %s, "attendance_percentage": 90}' % literal)
            self.assertEqual(response.status_code, 400, literal)

    def test_huge_integer_returns_400_not_a_server_crash(self):
        body = json.dumps({"study_hours": 10**400, "attendance_percentage": 90})
        self.assertEqual(self.post_raw(body).status_code, 400)

    # --- request format -------------------------------------------------------
    def test_wrong_content_type_is_415(self):
        response = self.post_raw('{"study_hours": 6, "attendance_percentage": 90}', content_type="text/plain")
        self.assertEqual(response.status_code, 415)

    def test_malformed_json_and_non_object_json_are_400(self):
        self.assertEqual(self.post_raw('{"study_hours": 6,').status_code, 400)
        self.assertEqual(self.post_raw("[1, 2]").status_code, 400)

    def test_wrong_route_and_wrong_method_return_json(self):
        not_found = self.client.post("/predic", json={})
        self.assertEqual(not_found.status_code, 404)
        self.assertIn("errors", not_found.get_json())
        wrong_method = self.client.get("/predict")
        self.assertEqual(wrong_method.status_code, 405)
        self.assertIn("errors", wrong_method.get_json())


class ModelPersistenceTests(unittest.TestCase):
    def test_saved_and_loaded_model_gives_identical_predictions(self):
        """Exercise 1 in test form: dump + load works without re-training."""
        samples = pd.DataFrame(
            {"study_hours": [0.5, 3, 5, 7, 10], "attendance_percentage": [55, 70, 85, 90, 100]}
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "copy.joblib"
            joblib.dump(api.model, path)
            reloaded = joblib.load(path)
        np.testing.assert_array_equal(api.model.predict(samples), reloaded.predict(samples))
        np.testing.assert_allclose(api.model.predict_proba(samples), reloaded.predict_proba(samples))


if __name__ == "__main__":
    unittest.main(verbosity=2)
