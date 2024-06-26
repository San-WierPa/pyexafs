import os
import unittest

import numpy as np
from larch import Group

from pyexafs.pyexafs import PyExafs

# Define the path to criteria.json globally
current_dir = os.path.dirname(os.path.abspath(__file__))
quality_criteria_json = os.path.join(
    current_dir, "..", "src", "pyexafs", "criteria.json"
)


class TestPyExafs(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.measurement_data = np.array(
            [[7100, 7101, 7102, 7103, 7104], [0.1, 0.2, 0.3, 0.4, 0.5]]
        )
        self.pyexafs = PyExafs(quality_criteria_json, verbose=False)

    def test_shorten_data(self):
        self.pyexafs.data = Group()
        self.pyexafs.data.energy = [1, 2, 3]
        self.pyexafs.data.mu = [4, 5, 6]
        self.pyexafs.data.flat = [7, 8, 9]
        self.pyexafs.data.pre_edge = [10, 11, 12]
        self.pyexafs.data.post_edge = [13, 14, 15]
        self.pyexafs.data.k = [16, 17, 18]
        self.pyexafs.data.chi = [19, 20, 21]
        self.pyexafs.shorten_data()
        self.assertEqual(self.pyexafs.data.energy, [1, 2])
        self.assertEqual(self.pyexafs.data.mu, [4, 5])
        self.assertEqual(self.pyexafs.data.flat, [7, 8])
        self.assertEqual(self.pyexafs.data.pre_edge, [10, 11])
        self.assertEqual(self.pyexafs.data.post_edge, [13, 14])
        self.assertEqual(self.pyexafs.data.k, [16, 17])
        self.assertEqual(self.pyexafs.data.chi, [19, 20])

    def test_clip_data(self):
        data = np.array([-10, -5, 0, 5, 10])
        clipped_data = self.pyexafs.clip_data(data, -5, 5)
        np.testing.assert_array_equal(clipped_data, [-5, -5, 0, 5, 5])

    def test_load_data(self):
        self.pyexafs.load_data(self.measurement_data)
        self.assertEqual(
            self.pyexafs.data.energy.tolist(), [7100, 7101, 7102, 7103, 7104]
        )
        self.assertEqual(self.pyexafs.data.mu.tolist(), [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_find_e0(self):
        # Using test data with a clear edge to avoid issues with max()
        energy = np.array([7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109])
        mu = np.array([0.1, 0.15, 0.25, 0.45, 0.7, 0.85, 0.9, 0.92, 0.93, 0.94])
        e0 = self.pyexafs.find_e0(energy, mu)
        self.assertEqual(e0, 7103)

    def test_check_edge_step(self):
        self.pyexafs.data = Group()
        self.pyexafs.data.edge_step = 0.5
        self.pyexafs.quality_criteria_sample = {"edge step": {"min": 0.4, "max": 0.6}}
        result = self.pyexafs.check_edge_step()
        self.assertTrue(result[0])
        self.assertEqual(result[1], 0.5)

    def test_check_energy_resolution(self):
        self.pyexafs.data = Group()
        self.pyexafs.data.energy = [7100, 7101]
        self.pyexafs.quality_criteria_sample = {
            "energy resolution": {"min": 0.9, "max": 1.1}
        }
        result = self.pyexafs.check_energy_resolution()
        self.assertTrue(result[0])
        self.assertEqual(result[1], 1)


if __name__ == "__main__":
    unittest.main()
