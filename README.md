# laplace_approximation_dist

A package for computing Laplace approximations to log probability densities in Python. The Laplace approximation computes a normal distribution which approximates the distribution of interest. The user sends in a log-probability function with optional optimization/second derivative information, and receives a scipy multivariate_normal which approximates the target distribution.

# How do I use this package?

In order to compute the laplace approximation to the desired distribution, you will need to provide the log-probability function along with (optionally) a function to compute the Hessian of your log-probability. If you do not provide a method to compute the Hessian, it will be computed numerically using a finite difference method from numdifftools.

## Simplest possible example: Normal approximations to normal distributions

### One-dimensional example

```
import numpy as np
from scipy.stats import norm, multivariate_normal, beta
from matplotlib import pyplot as plt
import seaborn as sns
from laplace_approximation_dist.approx import approx_dist
from functools import partial
from scipy.stats import binom

true_dist = norm(0, 1)
x0 = np.array([1])
result = approx_dist(true_dist.logpdf, x0)
plt.title('Samples from true and approximate distributions')
sns.distplot(true_dist.rvs(5000), label='True')
sns.distplot(result.rvs(5000), label='Approximate')
plt.legend()
plt.show()
```

### Multivariate normal

```
x0 = np.array([1,2])
true_dist = multivariate_normal(np.array([0, 0]), [[2.0, 0.8], [0.8, 0.5]])
result = approx_dist(true_dist.logpdf, x0)
print(result.mean)
print(result.cov)
true_samples = true_dist.rvs(10000)
approx_samples = result.rvs(10000)
plt.title('Samples from true and approximate distributions')
plt.scatter(approx_samples[:,0], approx_samples[:,1], label='Approximate', marker='.', alpha=.1, color='blue')
plt.scatter(true_samples[:,0], true_samples[:,1], label='True', marker='.', alpha=.1, color='red')
plt.legend()
plt.show()
```
## Approximating a beta distribution

```
x0 = np.array([0.5])
true_dist = beta(3, 12)
result = approx_dist(true_dist.logpdf, x0)
plt.title('Samples from true and approximate distributions')
sns.distplot(true_dist.rvs(5000), label='True')
sns.distplot(result.rvs(5000), label='Approximate')
plt.legend()
plt.show()
```

## Bayesian inference: Binomial proportion inference

```
x0 = np.array([0.25])
def posterior_logpdf(v):
    if 0 < v < 1:
        return binom.logpmf(200, 300, v)    
    return -np.inf
approx = approx_dist(posterior_logpdf, x0)
sns.distplot(beta(201, 101).rvs(5000), label='Conjugate prior')
sns.distplot(approx.rvs(5000), label='Laplace approxiation')
plt.legend()
plt.show()
```

## Bayesian inference: Linear regression

```
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
```

```
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
```

```
sns.distplot(samples[:,2])
plt.title('Posterior of residual noise')
plt.axvline(TRUE_S, label='True noise', color='orange')
plt.legend()
plt.show()
```

```
plt.title('Regression lines drawn from posterior')
for a_sample, b_sample, _ in samples:
    plt.plot(x, a_sample + b_sample*x, color='orange', alpha=0.1)
plt.scatter(x, y, alpha=0.1)
plt.show()
```

# How does the Laplace Approximation work?

The Laplace approximation computes a normal distribution which approximates the distribution of interest.  It does so by centering the approximation around the mode of the target distribution, and using the second derivative at the mode to find an approximation with the right curvature.

We can get some intuition by looking at the 1-dimensional case. This explanation was adapted from [these lecture notes](http://www2.stat.duke.edu/~st118/sta250/laplace.pdf).

Let h(θ) be the density of interest. h(θ) may not be normal, but we expect that it "looks normal" - the majority of the mass is in a dense region around the mode. 

Let q(θ) be the log density. That is:

q(θ) = log h(θ)

We'd like to find an approximation to q(θ), which will let us approximate h(θ). If the distribution is sharply peaked around the mode, θₘₐₓ, we can expect to approximate the distribution well if we only approximate the region around θₘₐₓ. To do so, we'll construct a second-order Taylor expansion around the mode:

q(θ) ≈ q(θₘₐₓ) + ½(θ - θₘₐₓ)² q''(θₘₐₓ) = const - (θ - a)² / (2b²)

Note that θₘₐₓ is the maximum by definition, so we've dropped the first-order term from the Taylor expansion (since q'(θₘₐₓ) = 0).

The form above bears a resemblance to a log normal distribution, where we have substituted

a = θₘₐₓ
b = (-q''(θₘₐₓ))⁻¹

So that q(θ) ≈ LogNorm(a, b²). Then we can approximate the original density with a normal distribution 

h(θ) ≈ N(a, b²)

In the multidimensional case, θₘₐₓ is the mean vector of a multivariate normal, and the negative-inverse hessian matrix gives the covariance matrix (the negative-inverse hessian here acting as the multidimensional analogue of the negative-reciprocal second derivative).

I found the following resources to be useful reading:

http://www2.stat.duke.edu/~st118/sta250/laplace.pdf

http://www.sumsar.net/blog/2013/11/easy-laplace-approximation/
