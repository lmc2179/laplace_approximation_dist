import numpy as np
from scipy.stats import norm, multivariate_normal
from laplace_approximation_dist.approx import approx_dist

def test_fxn(x):
	return x[0]**2+x[1]**2

x0 = np.array([1,2])
result = approx_dist(multivariate_normal(np.array([0, 0]), np.eye(2)).logpdf, x0)
print(result.cov)