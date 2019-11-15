# laplace_approximation_dist
laplace approximation to log probabilities in python

1D case:

q(θ) = log h(θ)
q(θ) ≈ q(θₘₐₓ) + ½(θ - θₘₐₓ)² q''(θₘₐₓ) = const - (θ - a)² / (2b²)

If we rewrite

a = θₘₐₓ
b = (-q''(θₘₐₓ))⁻¹

Then we can approximate the original density with a normal distribution 

h(θ) ≈ N(a, b²)

In the multidimensional case, θₘₐₓ is the mean vector of a Multivariate normal, and the negative-inverse hessian matrix gives the covariance matrix.

http://www2.stat.duke.edu/~st118/sta250/laplace.pdf
http://www.sumsar.net/blog/2013/11/easy-laplace-approximation/
