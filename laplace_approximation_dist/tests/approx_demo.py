import numpy as np
from scipy.stats import norm, multivariate_normal, beta
from matplotlib import pyplot as plt
import seaborn as sns
from laplace_approximation_dist.approx import approx_dist

# x0 = np.array([1,2])
# result = approx_dist(multivariate_normal(np.array([0, 0]), np.eye(2)).logpdf, x0)
# print(result.cov)

# x0 = np.array([1])
# true_dist = norm(0, 1)
# result = approx_dist(true_dist.logpdf, x0)
# sns.distplot(true_dist.rvs(5000))
# sns.distplot(result.rvs(5000))
# plt.show()

x0 = np.array([0.5])
true_dist = beta(15, 15)
result = approx_dist(true_dist.logpdf, x0)
sns.distplot(true_dist.rvs(5000))
sns.distplot(result.rvs(5000))
plt.show()