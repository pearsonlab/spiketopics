"""
pelt.py: a simple implementation of the PELT algorithm.
"""

from __future__ import division
import numpy as np
import scipy.stats as stats
from scipy.special import logit, expit
from scipy.stats import bernoulli
from numba import jit

@jit("float64(float64[:, :], float64, int64, int64)", nopython=True)
def base_LL(LL, theta, t1, t2):
    """
    For the closed interval [t1, t2], calculate the log likelihood of the
    data for z = 0.
    """
    bLL = 0
    for t in xrange(t1, t2 + 1):
        bLL += LL[t, 0]
    return bLL

@jit("float64(float64[:, :], float64, int64, int64)", nopython=True)
def kappa(LL, theta, t1, t2):
    """
    Calculate the differential log likelihood for assigning count data to
    state 1 vs state 0.
    LL is a two-column array, with each column containing the log likelihood
    of data observed in each time bin for z = 0 or 1, depending on column.
    theta is the prior parameter: p(z = 1) = theta
    """
    LLdiff = 0.
    for t in xrange(t1, t2 + 1):
        LLdiff += LL[t, 1] - LL[t, 0]
    return LLdiff + np.log(theta / (1 - theta))

@jit("float64(float64[:, :], float64, int64, int64)", nopython=True)
def C(LL, theta, t1, t2):
    """
    Calculate the cost function for data in the closed interval [t1, t2].
    """
    kap = kappa(LL, theta, t1, t2)
    return -(base_LL(LL, theta, t1, t2) + np.logaddexp(0, kap))

@jit
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
    K = 0.0  # bound constant for objective gain on changepoint insertion

    # initialize
    beta = alpha - np.log1p(-theta)
    R.add(-1)
    F[0] = -beta
    F[1:] = np.nan

    # CP will be an array indexed by times (tau)
    # each entry will be the changepoint added at that time (t' = new_cp)
    # we will recover the final CP list by recursing back up the tree,
    # since CP(tau) = CP(t') U {t'}
    CP = np.zeros(T).astype('int64')  # one entry

    # iterate
    for tau in xrange(T):
        mincost = np.inf
        for t in R:
            cost = F[t + 1] + C(LL, theta, t + 1, tau) + beta
            if cost < mincost:
                mincost = cost
                new_cp = t

        F[tau + 1] = mincost

        CP[tau] = new_cp

        Rnext = set({})
        for r in R:
            if F[r + 1] + C(LL, theta, r + 1, tau) + K < F[tau + 1]:
                Rnext.add(r)

        Rnext.add(tau)
        R = Rnext

    # extract changepoints
    # return sorted(CP[-1])
    cpset = set({})
    t = T - 1
    while t > 0:
        tprime = CP[t]
        cpset.add(tprime)
        t = tprime

    return sorted(cpset)

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
    Ctot = 0
    Htot = 0

    for tau in xrange(Ncp):
        run_start = cplist[tau] + 1

        if tau == Ncp - 1:
            run_end = T - 1
        else:
            run_end = cplist[tau + 1]

        kap = kappa(LL, theta, run_start, run_end)
        Ez = expit(kap)
        for t in xrange(run_start, run_end + 1):
            inferred[t] = Ez
        # inferred[run_start:(run_end + 1)] = Ez
        Ctot += C(LL, theta, run_start, run_end)
        Htot += bernoulli.entropy(Ez)

    return inferred

@jit
def find_changepoints_bs(LL, theta, alpha):
    """
    Find changepoints via binary segmentation. Same conventions as in
    find_changepoints.
    """
    T = LL.shape[0]  # number of time points
    segments = {(0, T)}  # initialize with a single segment
    CP = []  # list of changepoints
    beta = alpha - np.log1p(-theta)

    while len(segments):
        start, stop = segments.pop()
        new_cp = _find_cp_bs(LL[start:stop], beta, theta)

        if new_cp > 0:
            new_cp += start  # get cp relative to whole data set
            CP.append(new_cp)

            # New segments to search are [start, new_cp) and (new_cp, stop)
            # equivalent to slices [start, new_cp) and [new_cp + 1, stop)
            if new_cp - start > 2:
                segments.add((start, new_cp))
            if stop - new_cp > 3:
                segments.add((new_cp + 1, stop))

    return sorted(CP)

@jit("int64(float64[:, :], float64, float64)", nopython=True)
def _find_cp_bs(LL, beta, theta):
    """
    Find the best changepoint for bisecting the data range between start
    and stop.
    """
    T = LL.shape[0]

    # search all times, comparing cost of changepoint sequence to cost of
    # sequence taken as a whole

    C_nosplit = C(LL, theta, 0, T - 1)
    mincost = C_nosplit
    cp = -1

    for t in xrange(T):
        cost = C(LL, theta, 0, t) + C(LL, theta, t + 1, T - 1) + beta
        if cost < mincost:
            mincost = cost
            cp = t

    return cp
