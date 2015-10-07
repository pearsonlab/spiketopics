"""
pelt.py: a simple implementation of the PELT algorithm.
"""

import numpy as np
import scipy.stats as stats
from scipy.special import gammaln, logit
from numba import jit

# Define some helper functions
# @jit(nopython=True)
def grab_ss(counts, t1, t2):
    """
    Given start and end times, get sufficient statistics for data in the
    closed interval [t1, t2] from counts.
    """
    this_counts = counts[t1:(t2 + 1)]
    return np.sum(this_counts), len(this_counts)

# @jit("float64(int64[:], int64, int64, float64, float64)")
def base_LL(LL, theta, t1, t2):
    """
    For the closed interval [t1, t2], calculate the log likelihood of the
    data for z = 0.
    """
    Psi0 = np.sum(LL[t1:(t2 + 1), 0])
    return Psi0 + np.log(1 - theta)

# @jit("float64(int64, float64, float64, float64, float64)", nopython=True)
def kappa(LL, theta, t1, t2):
    """
    Calculate the differential log likelihood for assigning count data to
    state 1 vs state 0.
    LL is a two-column array, with each column containing the log likelihood
    of data observed in each time bin for z = 0 or 1, depending on column.
    theta is the prior parameter: p(z = 1) = theta
    """
    Psi0 = np.sum(LL[t1:(t2 + 1), 0])
    Psi1 = np.sum(LL[t1:(t2 + 1), 1])
    LLdiff = Psi1 - Psi0
    return LLdiff + logit(theta)

# @jit("float64(int64[:], int64, int64, float64, float64, float64)")
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
    R.add(0)
    F[0] = -beta

    # iterate
    for tau in xrange(1, T + 1):
        mincost = np.inf
        new_tau = 0
        for t in R:
            cost = F[t] + C(LL, theta, t + 1, tau) + beta
            if cost < mincost:
                mincost = cost
                new_tau = t

        F[tau] = mincost

        CP.add(new_tau)
        R = set({})
        for r in R:
            if F[r] + C(LL, theta, r + 1, tau) + K < F[tau]:
                R.add(r)

        R.add(tau)

    # extract changepoints
    return sorted(list(CP))

# @jit
def calc_state_probs(LL, theta, cplist):
    """
    Given a sorted iterable of changepoints (bin locations) and a maximum
    number of bins, return an array with each bin containing the probability
    of being in state 1.
    """
    Ncp = len(cplist)
    T = LL.shape[0]
    inferred = np.empty(T)

    for tau in xrange(Ncp):
        run_start = cplist[tau] + 1
        if tau == Ncp - 1:
            run_end = T
        else:
            run_end = cplist[tau + 1]

        kap = kappa(LL, theta, run_start, run_end)
        Ez = stats.logistic.cdf(kap)
        inferred[run_start:(run_end + 1)] = Ez

    return inferred