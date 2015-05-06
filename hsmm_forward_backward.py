"""
Contains code to run the forward-backward algorithm for inference in 
Hidden Semi-Markov Models.
"""
from __future__ import division
import numpy as np
from numba import jit, autojit

def fb_infer(logA, logpi, logpsi, dvec, logpd):
    """
    Implement the forward-backward inference algorithm.
    logA is a matrix of log transition probabilities that acts to the right:
    logpi is a matrix of log initial state probabilities
    new_state = A * old_state, so that columns of A sum to 1
    logpsi is the vector of log evidence: log p(y_t|z_t); 
        it does not need to be normalized, but the lack of normalization 
        will be reflected in logZ, such that the end result using the 
        given psi will be properly normalized when using the returned 
        value of Z
    dvec is an M x D vector of possible duration values for the hidden states
    logpd is an M x D vector of log probabilities for these durations
    """
    if np.any(logA > 0):
        raise ValueError('Transition matrix probabilities > 1')
    if np.any(logpi > 0):
        raise ValueError('Initial state probabilities > 1')

    # get shapes, preallocate arrays
    T = logpsi.shape[0]
    M, D = dvec.shape

    alpha = np.empty((T, M))
    alpha_star = np.empty((T, M))
    beta = np.empty((T, M))
    beta_star = np.empty((T, M))

    B = np.empty((T, M, D))
    cum_log_psi = np.empty((T, M))
    _calc_B(dvec, logpsi, B, cum_log_psi)
    del cum_log_psi  # free some memory

    # 
    _forward(alpha, alpha_star, logA, logpi, B, dvec, logpd)

@jit("void(int64[:], float64[:, :], float64[:, :, :], float64[:, :])")
def _calc_B(dvec, logpsi, B, cum_log_psi):
    """
    Calculate the logged version of B. If logpsi (T x M) is the log 
    probability of observing y_t given state M, then

    B_{tid} = \sum_{t' = t - d + 1}^{t} logpsi_{t'}(i)
    """
    T, M, D = B.shape

    # calculate cumulative sum of evidence
    for m in xrange(M):
        cum_log_psi[0, m] = logpsi[0, m]
        for t in xrange(1, T):
            cum_log_psi[t, m] = cum_log_psi[t - 1, m] + logpsi[t, m]

    # calculate B
    for t in xrange(T):
        for m in xrange(M):
            for i, d in enumerate(dvec):
                start = max(0, t - d + 1)
                if start > 0:
                    B[t, m, i] = cum_log_psi[t, m] - cum_log_psi[start - 1, m]
                else:
                    B[t, m, i] = cum_log_psi[t, m] 

@jit("void(float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:, :, :], int64[:], float64[:, :])", nopython=True)
def _forward(a, astar, A, pi, B, dvec, D):
    """
    Implement foward part of forward algorithm in log space.
    All inputs are logged, including A and pi.
    """
    T, M = a.shape

    for t in xrange(T):
        for m in xrange(M):
            # calculate a^*[t, m]
            if t == 0:
                astar[t, m] = pi[m]
            else:
                astar[t, m] = -np.inf
                for j in xrange(M):
                    astar[t, m] = np.logaddexp(A[m, j] + a[t, j], astar[t, m])

            # calculate a[t, m]
            a[t, m] = -np.inf
            for didx, d in enumerate(dvec):
                if d <= t:
                    a[t, m] = np.logaddexp(B[t, m, didx] + D[m, didx] + 
                        astar[t - d, m], a[t, m])







