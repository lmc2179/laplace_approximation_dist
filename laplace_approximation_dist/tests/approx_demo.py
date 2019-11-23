import numpy as np
from scipy.stats import norm, multivariate_normal, beta
from matplotlib import pyplot as plt
import seaborn as sns
from laplace_approximation_dist.approx import approx_dist
from functools import partial
from scipy.stats import binom

# x0 = np.array([1,2])
# true_dist = multivariate_normal(np.array([0, 0]), [[2.0, 0.8], [0.8, 0.5]])
# result = approx_dist(true_dist.logpdf, x0)
# print(result.mean)
# print(result.cov)
# true_samples = true_dist.rvs(10000)
# approx_samples = result.rvs(10000)
# plt.title('Samples from true and approximate distributions')
# plt.scatter(approx_samples[:,0], approx_samples[:,1], label='Approximate', marker='.', alpha=.1, color='blue')
# plt.scatter(true_samples[:,0], true_samples[:,1], label='True', marker='.', alpha=.1, color='red')
# plt.legend()
# plt.show()

# x0 = np.array([1])
# true_dist = norm(0, 1)
# result = approx_dist(true_dist.logpdf, x0)
# plt.title('Samples from true and approximate distributions')
# sns.distplot(true_dist.rvs(5000), label='True')
# sns.distplot(result.rvs(5000), label='Approximate')
# plt.legend()
# plt.show()

# x0 = np.array([0.5])
# true_dist = beta(3, 12)
# result = approx_dist(true_dist.logpdf, x0)
# plt.title('Samples from true and approximate distributions')
# sns.distplot(true_dist.rvs(5000), label='True')
# sns.distplot(result.rvs(5000), label='Approximate')
# plt.legend()
# plt.show()

# Beta binomial - Bayesian inference
# x0 = np.array([0.25])
# def posterior_logpdf(v):
    # if 0 < v < 1:
        # return binom.logpmf(200, 300, v)    
    # return -np.inf
# approx = approx_dist(posterior_logpdf, x0)
# sns.distplot(beta(201, 101).rvs(5000), label='Conjugate prior')
# sns.distplot(approx.rvs(5000), label='Laplace approxiation')
# plt.legend()
# plt.show()

# Linear regression
def lin_regression_lnprob(v, x=None, y=None):
	a, b, s = v
	if s <= 0:
		return -np.inf
	y_hat = a + b*x
	lp = np.sum(norm.logpdf(y, y_hat, s))
	return lp

TRUE_A = 0.1
TRUE_B = -10
TRUE_S = 1.3
x = np.linspace(0, 1, 10000)
y = TRUE_A + TRUE_B*x + np.random.normal(0, TRUE_S, 10000)
L = partial(lin_regression_lnprob, x=x, y=y)
	
result = approx_dist(L, np.array([0, 0, 1]))
samples = result.rvs(1000)
plt.scatter(samples[:,0], samples[:,1], alpha=0.1)
plt.scatter([np.mean(samples[:,0])], [np.mean(samples[:,1])], marker='x', color='red', label='Posterior mean')
plt.title('Posterior samples for intercept and coefficient of linear model')
plt.xlabel('Intercept')
plt.ylabel('Coefficient')
plt.axvline(TRUE_A, color='orange', label='True intercept')
plt.axhline(TRUE_B, label='True Coefficient')
plt.legend()
plt.show()

sns.distplot(samples[:,2])
plt.title('Posterior of residual noise')
plt.axvline(TRUE_S, label='True noise', color='orange')
plt.legend()
plt.show()

plt.title('Regression lines drawn from posterior')
for a_sample, b_sample, _ in samples:
    plt.plot(x, a_sample + b_sample*x, color='orange', alpha=0.1)
plt.scatter(x, y, alpha=0.1)
plt.show()