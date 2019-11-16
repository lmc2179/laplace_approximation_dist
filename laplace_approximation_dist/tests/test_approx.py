import unittest
import numpy as np
from scipy.stats import norm, beta
from laplace_approximation_dist.approx import approx_dist
from functools import partial

class TestApproximation(unittest.TestCase):
    def test_norm_approx_1d(self):
        x0 = np.array([1])
        true_dist = norm(0, 1)
        approx = approx_dist(true_dist.logpdf, x0)
        self.assertAlmostEqual(float(approx.mean[0]), 0)
        self.assertAlmostEqual(float(approx.cov[0]), 1)
        
    def test_beta_approx_1d(self):
        x0 = np.array([0.9])
        true_dist = beta(1000, 1000)
        approx = approx_dist(true_dist.logpdf, x0)
        self.assertAlmostEqual(float(approx.mean[0]), true_dist.mean())
        self.assertAlmostEqual(float(approx.cov[0]), true_dist.var(), delta=1e-5)