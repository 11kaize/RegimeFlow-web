"""Regression checks for preprocessing, isolated from learned forecast quality.

Run: python -m unittest discover -s tests -p test_web_preprocessing.py
"""
import unittest
from unittest.mock import patch

import numpy as np

from web.backend.engine_onnx import BLRPriorNP, RegimeFlowEngineONNX, StdScalerNP


class TemporalScalerTests(unittest.TestCase):
    def test_statistics_are_per_batch_and_channel_over_time(self):
        values = np.array([[[0., 10.], [4., 30.], [8., 50.]]])
        loc, scale = StdScalerNP(minimum_scale=1.).fit_transform(values)
        self.assertEqual(loc.shape, (1, 1, 2))
        np.testing.assert_allclose(loc, [[[4., 30.]]])
        np.testing.assert_allclose(scale, [[[np.sqrt(32. / 3.), np.sqrt(800. / 3.)]]])

    def test_mask_excludes_missing_observations(self):
        values = np.array([[[2.], [1000.], [6.]]])
        observed = np.array([[[1.], [0.], [1.]]])
        loc, scale = StdScalerNP(minimum_scale=1.).fit_transform(values, observed)
        np.testing.assert_allclose(loc, [[[4.]]])
        np.testing.assert_allclose(scale, [[[2.]]])

    def test_minimum_applies_to_standard_deviation(self):
        values = np.array([[[0.], [4.]]])
        _, scale = StdScalerNP(minimum_scale=4.).fit_transform(values)
        np.testing.assert_allclose(scale, [[[4.]]])


class PredictionPreprocessingTests(unittest.TestCase):
    def make_engine(self):
        # No model files needed: a zero vector field exposes the exact source
        # initialization and denormalization, rather than forecast accuracy.
        engine = object.__new__(RegimeFlowEngineONNX)
        engine.context_length = 96
        engine.prediction_length = 256
        engine.use_condition = False
        engine.cond_dim = 128
        engine.denoise_steps = 4
        engine.flow_sigma = .001
        engine.scaler = StdScalerNP(minimum_scale=1.)
        engine.prior = BLRPriorNP({'alpha': 1., 'beta': 10., 'noise_scale': .1})
        captured = []

        def zero_velocity(t, x, condition):
            captured.append(x.copy())
            return np.zeros_like(x)

        engine._backbone_velocity = zero_velocity
        return engine, captured

    def test_context_is_scaled_without_centering_and_future_is_denormalized(self):
        engine, captured = self.make_engine()
        with patch('numpy.random.randn', side_effect=lambda *shape: np.zeros(shape)):
            prediction = engine.predict([0., 4.] * 48)
        # loc=2, scale=2; original RegimeFlow keeps context / scale, while
        # centering the prior observations and adding loc back to predictions.
        np.testing.assert_allclose(captured[0][0, :2, 0], [0., 2.])
        np.testing.assert_allclose(captured[0][0, 96:, 0], 0., atol=1e-7)
        np.testing.assert_allclose(prediction, 2., atol=1e-7)

    def test_flow_noise_is_small_and_separate_from_blr_noise(self):
        engine, captured = self.make_engine()
        with patch('numpy.random.randn', side_effect=lambda *shape: np.ones(shape)):
            prediction = engine.predict([5.] * 96)
        np.testing.assert_allclose(captured[0][0, :96, 0], 5.001, atol=1e-6)
        np.testing.assert_allclose(captured[0][0, 96:, 0], .101, atol=1e-7)
        np.testing.assert_allclose(prediction, 5.101, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
