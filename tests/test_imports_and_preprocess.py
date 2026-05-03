import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class ProjectSmokeTests(unittest.TestCase):
    def test_app_import_does_not_load_tensorflow_model(self):
        import app

        self.assertEqual(app.app.title, "Concrete Crack Classification API")

    def test_preprocess_image_returns_model_batch_shape(self):
        from src.inference import preprocess_image

        image = np.full((12, 18, 3), 128, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), image))

            batch = preprocess_image(str(image_path))

        self.assertEqual(batch.shape, (1, 200, 200, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertGreaterEqual(float(batch.min()), 0.0)
        self.assertLessEqual(float(batch.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
