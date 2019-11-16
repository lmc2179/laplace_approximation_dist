import unittest
import numpy as np
from scipy.stats import norm, beta, binom
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
        
    def test_binomial_observation_posterior(self):
        x0 = np.array([0.25])
        def posterior_logpdf(v):
            if 0 < v < 1:
                return binom.logpmf(2000, 3000, v)    
            return -np.inf
        approx = approx_dist(posterior_logpdf, x0)
        true_dist = beta(2001, 1001)
        self.assertAlmostEqual(float(approx.mean[0]), true_dist.mean(), delta=1e-3)
        self.assertAlmostEqual(float(approx.cov[0]), true_dist.var(), delta=1e-5)
        
    def test_linear_regression_posterior(self):
        def lin_regression_lnprob(v, x=None, y=None):
            a, b, s = v
            if s <= 0:
                return -np.inf
            y_hat = a + b*x
            lp = np.sum(norm.logpdf(y, y_hat, s))
            return lp

        x = np.linspace(0, 5000, 1000)
        y = 5 - 10*x + np.random.normal(0, 2, 1000)
        L = partial(lin_regression_lnprob, x=x, y=y)
            
        approx = approx_dist(L, np.array([0, 0, 1]))
        self.assertAlmostEqual(float(approx.mean[0]), 5, delta=1e-1)
        self.assertAlmostEqual(float(approx.mean[1]), -10, delta=1e-1)
        self.assertAlmostEqual(float(approx.mean[2]), 2, delta=1e-1)