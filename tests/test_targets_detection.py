import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR / "src"))

from data.targets_detection import (  # noqa: E402
    anatomical_class,
    apply_hu_window,
    build_detection_target,
    letterbox_pair,
)


class TargetsDetectionTest(unittest.TestCase):
    def test_taxonomy(self):
        self.assertEqual(anatomical_class(1), 1)
        self.assertEqual(anatomical_class(10), 1)
        self.assertEqual(anatomical_class(11), 2)
        self.assertEqual(anatomical_class(20), 2)
        self.assertEqual(anatomical_class(21), 3)
        self.assertEqual(anatomical_class(30), 3)
        with self.assertRaises(ValueError):
            anatomical_class(31)

    def test_fragments_of_same_bone_form_one_box(self):
        label = np.zeros((20, 30), dtype=np.int16)
        label[2:6, 3:8] = 11
        label[10:15, 20:27] = 12
        label[5:12, 10:14] = 21

        target = build_detection_target(label, "synthetic", 4, min_pixels=1)

        np.testing.assert_array_equal(target["labels"], [2, 3])
        np.testing.assert_array_equal(target["boxes"][0], [3, 2, 27, 15])
        np.testing.assert_array_equal(target["boxes"][1], [10, 5, 14, 12])
        np.testing.assert_array_equal(target["instance_ids"], [11, 12, 21])

    def test_hu_window_is_clipped_and_normalized(self):
        values = np.array([-1000, -500, 400, 1300, 2000], dtype=np.float32)
        result = apply_hu_window(values, center=400, width=1800)
        np.testing.assert_allclose(result, [0, 0, 0.5, 1, 1])

    def test_letterbox_preserves_labels(self):
        image = np.ones((10, 20), dtype=np.float32)
        label = np.zeros((10, 20), dtype=np.int16)
        label[2:8, 4:16] = 1
        image_out, label_out, metadata = letterbox_pair(image, label, output_size=40)

        self.assertEqual(image_out.shape, (40, 40))
        self.assertEqual(label_out.shape, (40, 40))
        self.assertEqual(metadata["offset_y"], 10)
        self.assertEqual(set(np.unique(label_out)), {0, 1})


if __name__ == "__main__":
    unittest.main()
