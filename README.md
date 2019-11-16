# laplace_approximation_dist

A package for computing Laplace approximations to log probability densities in Python.

# How do I use this package?

In order to compute the laplace approximation to the desired distribution, you will need to provide the log-probability function along with (optionally) a function to compute the Hessian of your log-probability. If you do not provide a method to compute the Hessian, it will be computed numerically using a finite difference method.

# How does the Laplace Approximation work?

The Laplace approximation computes a normal distribution which approximates the distribution of interest.  It does so by centering the approximation around the mode of the target distribution, and using the second derivative at the mode to find an approximation with the right curvature.

We can get some intuition by looking at the 1-dimensional case. This explanation was adapted from [these lecture notes](http://www2.stat.duke.edu/~st118/sta250/laplace.pdf).

Let h(θ) be the density of interest, and let q(θ) be the log density. That is:

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
