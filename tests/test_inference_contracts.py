"""Boundary and source-prior checks; actual-model validation is a separate script."""
import unittest
import numpy as np
from web.backend.inference_config import load_config, resolve_condition, transform_context
from web.backend.engine_onnx import BLRPriorNP


class InputContracts(unittest.TestCase):
    def test_condition_fallback_and_nonoscillating_period(self):
        for pattern, period in [(None, 2), (6, 2), (3, 0), (3, float('nan')), (3, -1)]:
            self.assertEqual(resolve_condition(pattern, period), (0, 0.))
        self.assertEqual(resolve_condition(2, None), (2, 0.))
        self.assertEqual(resolve_condition(3, 25), (3, 25.))

    def test_custom_bounds_use_history_and_support_signed_constant_data(self):
        values, lo, span = transform_context([-4, 0, 4])
        np.testing.assert_allclose(values, [0, .5, 1])
        self.assertEqual((lo, span), (-4, 8))
        values, lo, span = transform_context([-5, -5])
        np.testing.assert_array_equal(values, [0, 0])
        self.assertEqual((lo, span), (-5, 0))

    def test_dataset_filtering_and_invalid_input(self):
        values, lo, span = transform_context([-1, 2], [0, 4])
        np.testing.assert_allclose(values, [0, .5])
        for values in ([], [1, float('nan')], [float('inf')]):
            with self.assertRaises(ValueError): transform_context(values)
        with self.assertRaises(ValueError): transform_context([2], [0, 1])

    def test_all_regimes_have_continuous_prior_and_controlled_noise(self):
        prior = BLRPriorNP(); past = np.arange(96); future = np.arange(96, 352)
        values = np.sin(past/10) + past*.002
        sizes = [1, 4, 4, 7, 3, 3]
        for pattern in range(6):
            period = 24 if pattern == 3 else 0
            self.assertEqual(prior.basis(past, past, pattern, period).shape, (96, sizes[pattern]))
            mean, variance = prior.mean(past, values, future, pattern, period)
            self.assertAlmostEqual(mean[0], values[-1], places=12)
            self.assertTrue(np.isfinite(mean).all())
            self.assertTrue((variance >= prior.min_variance).all())
            sample = prior.generate(past, values, future, pattern, period, noise=np.ones(256))
            np.testing.assert_allclose(sample, mean + load_config()['noise_scale'], atol=1e-6)

    def test_polynomial_past_and_future_scales_are_separate(self):
        prior = BLRPriorNP()
        pp=prior.basis(np.arange(96), np.arange(96), 4, 0)
        pf=prior.basis(np.arange(96,352), np.arange(96), 4, 0)
        np.testing.assert_allclose(pp[[0,-1],1], [-1, 0])
        np.testing.assert_allclose(pf[[0,-1],1], [1/256, 1])


if __name__ == '__main__': unittest.main()
