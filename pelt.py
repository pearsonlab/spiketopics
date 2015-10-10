"""
pelt.py: a simple implementation of the PELT algorithm.
"""

import numpy as np
import scipy.stats as stats
from scipy.special import gammaln, logit
from numba import jit

@jit("float64(float64[:, :], float64, int64, int64)", nopython=True)
def base_LL(LL, theta, t1, t2):
    """
    For the closed interval [t1, t2], calculate the log likelihood of the
    data for z = 0.
    """
    return np.sum(LL[t1:(t2 + 1), 0])

@jit("float64(float64[:, :], float64, int64, int64)", nopython=True)
def kappa(LL, theta, t1, t2):
    """
    Calculate the differential log likelihood for assigning count data to
    state 1 vs state 0.
    LL is a two-column array, with each column containing the log likelihood
    of data observed in each time bin for z = 0 or 1, depending on column.
    theta is the prior parameter: p(z = 1) = theta
    """
    LLdiff = 0
    for t in xrange(t1, t2 + 1):
        LLdiff += LL[t, 1] - LL[t, 0]
    return LLdiff + np.log(theta / (1 - theta))

@jit("float64(float64[:, :], float64, int64, int64)", nopython=True)
def C(LL, theta, t1, t2):
    """
    Calculate the cost function for data in the closed interval [t1, t2].
    """
    kap = kappa(LL, theta, t1, t2)
    return -(base_LL(LL, theta, t1, t2) + kap + np.logaddexp(0, -kap))

# @jit
def find_changepoints(LL, theta, alpha):
    """
    Given the an array of log likelihoods at each time (row = time,
    col = z value), a prior on z (p(z = 1) = theta), and a prior parameter
    on number of changepoints (log p(m) ~ -alpha),
    find changepoints in the data. Points mark the beginning of a new section,
    (meaning: the changepoint bin is *included*).
    """
    # allocate containers
    T = LL.shape[0]  # number of time points
    F = np.empty(T + 1)  # F(t) = minimum cost for all data up to time t
    R = set({})  # set of times over which to search
    CP = set({})  # list of changepoints (ordered)
    K = 0.0  # bound constant for objective gain on changepoint insertion

    # initialize
    beta = alpha - np.log1p(-theta)
    R.add(-1)
    F[0] = -beta

    # iterate
    for tau in xrange(T):
        mincost = np.inf
        new_tau = 0
        for t in R:
            cost = F[t + 1] + C(LL, theta, t + 1, tau) + beta
            if cost < mincost:
                mincost = cost
                new_tau = t

        F[tau + 1] = mincost

        CP.add(new_tau)
        Rnext = set({})
        for r in R:
            if F[r + 1] + C(LL, theta, r + 1, tau) + K < F[tau + 1]:
                Rnext.add(r)

        Rnext.add(tau)
        R = Rnext

    # extract changepoints
    # return sorted(list(CP))
    return sorted(list(CP))

@jit
def calc_state_probs(LL, theta, cplist):
    """
    Given a sorted iterable of changepoints (bin locations) and a maximum
    number of bins, return an array with each bin containing the probability
    of being in state 1.
    """
    Ncp = len(cplist)
    T = LL.shape[0]
    inferred = np.zeros(T)

    for tau in xrange(Ncp):
        # if cplist[tau] > 0:
        run_start = cplist[tau] + 1
        # else:
        #     run_start = 0

        if tau == Ncp - 1:
            run_end = T
        else:
            run_end = cplist[tau + 1]

        kap = kappa(LL, theta, run_start, run_end)
        Ez = stats.logistic.cdf(kap)
        inferred[run_start:(run_end + 1)] = Ez

    return inferred