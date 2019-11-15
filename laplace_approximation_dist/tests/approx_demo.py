import numpy as np
from scipy.stats import norm, multivariate_normal, beta
from matplotlib import pyplot as plt
import seaborn as sns
from laplace_approximation_dist.approx import approx_dist
from functools import partial

# x0 = np.array([1,2])
# result = approx_dist(multivariate_normal(np.array([0, 0]), np.eye(2)).logpdf, x0)
# print(result.cov)

# x0 = np.array([1])
# true_dist = norm(0, 1)
# result = approx_dist(true_dist.logpdf, x0)
# sns.distplot(true_dist.rvs(5000))
# sns.distplot(result.rvs(5000))
# plt.show()

# x0 = np.array([0.5])
# true_dist = beta(10, 15)
# result = approx_dist(true_dist.logpdf, x0)
# sns.distplot(true_dist.rvs(5000))
# sns.distplot(result.rvs(5000))
# plt.show()

# Beta binomial
# x0 = np.array([1., 1.])

# Linear regression
def lin_regression_lnprob(v, x=None, y=None):
	a, b, s = v
	if s <= 0:
		return -np.inf
	y_hat = a + b*x
	lp = np.sum(norm.logpdf(y, y_hat, s))
	return lp

x = np.linspace(0, 1, 1000)
y = .1 - 10*x + np.random.normal(0, 1, 1000)
L = partial(lin_regression_lnprob, x=x, y=y)
	
result = approx_dist(L, np.array([0, 0, 1]))
samples = result.rvs(1000)
sns.jointplot(samples[:,0], samples[:,1])
plt.show()
sns.distplot(samples[:,2])
plt.show()