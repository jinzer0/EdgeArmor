import unittest

from src.pipeline.inference_contract import (
    CLASSIFIER_INPUT_RESOLUTION,
    DEFAULT_FAKE_THRESHOLD,
    aggregate_face_predictions,
    build_extension_contract,
    build_failed_result,
    build_no_face_result,
    classifier_patch_count,
    compute_face_label,
)


class InferenceContractTests(unittest.TestCase):
    def test_build_no_face_result_preserves_expected_shape(self):
        result = build_no_face_result(num_detected_faces=3)

        self.assertEqual(result["status"], "no_face")
        self.assertIsNone(result["image_fake_prob"])
        self.assertEqual(result["image_pred_label"], "real")
        self.assertEqual(result["num_detected_faces"], 3)
        self.assertEqual(result["num_faces"], 0)
        self.assertEqual(result["faces"], [])

    def test_build_failed_result_can_include_failure_counts(self):
        result = build_failed_result(num_detected_faces=2, num_failed_faces=2)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["num_detected_faces"], 2)
        self.assertEqual(result["num_failed_faces"], 2)
        self.assertEqual(result["num_faces"], 0)
        self.assertEqual(result["faces"], [])

    def test_aggregate_face_predictions_builds_summary(self):
        face_predictions = [
            {"face_index": 0, "fake_prob": 0.25},
            {"face_index": 1, "fake_prob": 0.75},
        ]

        result = aggregate_face_predictions(
            face_predictions,
            fake_threshold=DEFAULT_FAKE_THRESHOLD,
            num_detected_faces=3,
            num_failed_faces=1,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["image_fake"], 1)
        self.assertEqual(result["image_pred_label"], "fake")
        self.assertEqual(result["num_detected_faces"], 3)
        self.assertEqual(result["num_failed_faces"], 1)
        self.assertEqual(result["num_faces"], 2)
        self.assertEqual(result["summary"]["selected_face_index"], 1)
        self.assertAlmostEqual(result["summary"]["max_fake_prob"], 0.75)
        self.assertAlmostEqual(result["summary"]["mean_fake_prob"], 0.5)
        self.assertAlmostEqual(result["summary"]["fake_threshold"], DEFAULT_FAKE_THRESHOLD)

    def test_extension_contract_matches_python_contract_values(self):
        contract = build_extension_contract()

        self.assertEqual(contract["classifier"]["inputSize"], CLASSIFIER_INPUT_RESOLUTION)
        self.assertEqual(contract["classifier"]["ifBoundaryLength"], classifier_patch_count())
        self.assertEqual(contract["classifier"]["fakeThreshold"], DEFAULT_FAKE_THRESHOLD)
        self.assertEqual(contract["selection"]["topK"], 3)

    def test_compute_face_label_respects_predicted_class_and_threshold(self):
        self.assertEqual(compute_face_label(0.9, 1), "fake")
        self.assertEqual(compute_face_label(0.4, 1), "real")
        self.assertEqual(compute_face_label(0.9, 0), "real")


if __name__ == "__main__":
    unittest.main()
