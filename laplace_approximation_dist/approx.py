import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from numdifftools import Hessian

def approx_dist(lnprob, v_init=None, v_max=None, hessian=None):
    """
    Construct a laplace approximation to the distribution with the given log-density.
    Arguments:
    -lnprob: The log-density which you would like to approximate. Should take a vector and return a real number.
    -v_init: The initial value at which to start the search for the mode of lnprob. If it is not given.
    -v_max: The mode of lnprob. If it is not provided, it will be calculated numerically.
    -hessian: A function which will compute the Hessian. If it is not given, it will be approximated numerically.
    Returns:
    -approximate distribution, a scipy.stats.multivariate_normal object
    """
    neg_lnprob = lambda v: -lnprob(v)
    if v_max is None and v_init is not None:
        result = minimize(neg_lnprob, v_init)
        x_max = result.x
    elif v_max is not None:
        x_max = v_max
    else:
        raise Exception('You must provide either an initial value at which to start the search for the mode (v_init) or the value of the mode (v_max)')
    if hessian is None:
        hess_calc = Hessian(lnprob)
    else:
        hess_calc = hessian
    h = hess_calc(x_max)
    dist = multivariate_normal(x_max, -np.linalg.inv(h))
    return dist