import unittest

import numpy as np

from distancias_fragmentos import measure_fragment_separations


class MeasureFragmentSeparationsTests(unittest.TestCase):
    spacing_xyz = (0.7, 1.3, 2.1)
    region_by_id = {1: "pelvis", 2: "pelvis"}

    def measure_pair(self, fragment_offset_zyx):
        volume = np.zeros((11, 11, 11), dtype=np.int16)
        volume[5, 5, 5] = 1
        volume[5 + fragment_offset_zyx[0],
               5 + fragment_offset_zyx[1],
               5 + fragment_offset_zyx[2]] = 2
        result = measure_fragment_separations(
            volume,
            self.spacing_xyz,
            region_by_id=self.region_by_id,
            main_id_by_region={"pelvis": 1},
        )
        return float(result.loc[result["fragment_id"] == 2, "distance_mm"].iloc[0])

    def test_face_edge_and_corner_contact_have_zero_distance(self):
        for offset in ((0, 0, 1), (0, 1, 1), (1, 1, 1)):
            with self.subTest(offset=offset):
                self.assertAlmostEqual(self.measure_pair(offset), 0.0)

    def test_axis_gap_uses_physical_spacing_and_array_axis_order(self):
        self.assertAlmostEqual(self.measure_pair((0, 0, 2)), self.spacing_xyz[0])
        self.assertAlmostEqual(self.measure_pair((0, 2, 0)), self.spacing_xyz[1])
        self.assertAlmostEqual(self.measure_pair((2, 0, 0)), self.spacing_xyz[2])

    def test_diagonal_gap_is_euclidean_between_voxel_cell_borders(self):
        expected = np.hypot(self.spacing_xyz[0], self.spacing_xyz[1])
        self.assertAlmostEqual(self.measure_pair((0, 2, 2)), expected)


if __name__ == "__main__":
    unittest.main()
