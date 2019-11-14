import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from numdifftools import Hessian

def approx_dist(lnprob, theta_init):
	neg_lnprob = lambda v: -lnprob(v)
	result = minimize(neg_lnprob, theta_init)
	# x_max, hess_inv = result.x, result.hess_inv
	x_max = result.x
	h = Hessian(lnprob)(x_max)
	print(h)
	dist = multivariate_normal(x_max, -np.linalg.inv(h))
	return dist