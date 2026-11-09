import unittest

import numpy as np

from opf_thickness_prior import apply_thickness_prior, fit_thickness_prior


class ThicknessPriorTests(unittest.TestCase):
    def test_learns_grouped_grid_without_reading_other_labels(self):
        layers = np.array(
            [
                [100, 0],
                [100, 0],
                [150, 0],
                [200, 0],
                [200, 0],
                [250, 250],
                [100, 100],
                [np.nan, np.nan],
            ]
        )
        density = np.where(layers > 0, 0.875, 0)
        density[-1] = np.nan
        groups = ["a", "a", "b", "c", "d", "e", "f", "holdout"]
        mask = np.array([True] * 7 + [False])
        prior = fit_thickness_prior(layers, density, groups, mask)
        self.assertEqual(prior.step_mm, 50)
        self.assertEqual(prior.fitting_structures, 6)
        self.assertAlmostEqual(prior.weight, 0.8)

        conversion = 1 + 0.845 * prior.density_g_cm3
        observed = np.array([[64, 0], [36, 65], [5, 0], [75, 0]]) * conversion
        result = apply_thickness_prior(observed, prior, 0.845)
        np.testing.assert_array_equal(result > 0, observed > 0)
        np.testing.assert_allclose(
            result / conversion, [[52.8, 0], [47.2, 53], [5, 0], [75, 0]]
        )

    def test_irregular_training_specimens_leave_delays_unchanged(self):
        layers = np.array([[71], [98], [134], [169], [213], [246]])
        prior = fit_thickness_prior(
            layers,
            np.full_like(layers, 0.87, dtype=float),
            list("abcdef"),
            np.ones(6, dtype=bool),
        )
        self.assertIsNone(prior)
        observed = np.array([[111, 0], [71, 91]])
        np.testing.assert_array_equal(
            apply_thickness_prior(observed, prior, 0.845), observed
        )
