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
    M, D = logpd.shape

    alpha = np.empty((T + 1, M))
    alpha_star = np.empty((T + 1, M))
    beta = np.empty((T + 1, M))
    beta_star = np.empty((T + 1, M))

    B = np.empty((T + 1, M, D))
    cum_log_psi = np.empty((T + 1, M))
    _calc_B(dvec, logpsi, B, cum_log_psi)
    del cum_log_psi  # free some memory

    # forward pass
    _forward(alpha, alpha_star, logA, logpi, B, dvec, logpd)

    # backward pass
    _backward(beta, beta_star, logA, B, dvec, logpd) 

    # calculate normalization constant
    logZ = _calc_logZ(alpha)

    # calculate posterior
    gamma = np.empty((T + 1, M))
    gamma_star = np.empty((T + 1, M))
    post = np.empty((T + 1, M))
    _calc_posterior(alpha, alpha_star, beta, beta_star, gamma, 
        gamma_star, post)
    del gamma, gamma_star

@jit("void(int64[:], float64[:, :], float64[:, :, :], float64[:, :])")
def _calc_B(dvec, logpsi, B, cum_log_psi):
    """
    Calculate the logged version of B. If logpsi (T x M) is the log 
    probability of observing y_t given state M, then

    B_{tid} = \sum_{t' = t - d + 1}^{t} logpsi_{t'}(i)

    Note: B and cum_log_psi have one more time index than logpsi for 
    easier handling of boundary conditions later on.
    """
    T, M, _ = B.shape

    # calculate cumulative sum of evidence
    for m in xrange(M):
        cum_log_psi[0, m] = 0
        for t in xrange(1, T):
            cum_log_psi[t, m] = cum_log_psi[t - 1, m] + logpsi[t - 1, m]

    # calculate B
    for t in xrange(T):
        for m in xrange(M):
            for ix, d in enumerate(dvec):
                start = max(1, t - d + 1)
                B[t, m, ix] = cum_log_psi[t, m] - cum_log_psi[start - 1, m]

@jit("void(float64[:, :], float64[:, :], float64[:, :], float64[:],  float64[:, :, :], int64[:], float64[:, :])", nopython=True)
def _forward(a, astar, A, pi, B, dvec, D):
    """
    Implement foward pass of forward-backward algorithm in log space.
    All inputs are logged, including A and pi.
    """
    T, M = a.shape

    for m in xrange(M):
        astar[0, m] = pi[m]
        a[0, m] = -np.inf

    for t in xrange(1, T):
        for m in xrange(M):
            # calculate a[t, m]
            a[t + 1, m] = -np.inf
            for ix, d in enumerate(dvec):
                if t - d > -1:
                    a[t + 1, m] = np.logaddexp(B[t, m, ix] + D[m, ix] + 
                        astar[t - d + 1, m], a[t, m])
                elif t - d == -1:
                    a[t, m] = np.logaddexp(B[t, m, ix] + D[m, ix] + 
                        pi[m], a[t, m])

        for m in xrange(M):
            # calculate a^*[t, m]
            astar[t, m] = -np.inf
            for j in xrange(M):
                astar[t, m] = np.logaddexp(A[m, j] + a[t, j], astar[t, m])


@jit("void(float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], int64[:], float64[:, :])", nopython=True)
def _backward(b, bstar, A, B, dvec, D):
    """
    Implement backward pass of forward-backward algorithm in log space.
    All inputs are logged, including A.
    """
    T, M = b.shape

    # initialize
    for m in xrange(M):
        b[-1, m] = 1
        bstar[-1, m] = 1

    for t in xrange(T - 2, -1, -1):
        for m in xrange(M):
            # calculate b^*[t, m]
            bstar[t, m] = -np.inf
            for didx, d in enumerate(dvec):
                if t + d < T:
                    bstar[t, m] = np.logaddexp(b[t + d, m] + 
                        B[t + d, m, didx] + D[m, didx], bstar[t, m])

        for m in xrange(M):
            # calculate b[t, m]
            b[t, m] = -np.inf
            for j in xrange(M):
                b[t, m] = np.logaddexp(bstar[t, j] + A[j, m], b[t, m])

@jit("float64(float64[:, :])", nopython=True)
def _calc_logZ(alpha):
    """
    Calculate the log of the partition function, given by summing
    p(z_{T - 1}, y_{0:T-1}) over all z_{T - 1}
    """
    T, M = alpha.shape
    logZ = -np.inf

    for m in xrange(M):
        logZ = np.logaddexp(alpha[-1, m], logZ)

    return logZ

@jit("void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :])")
def _calc_posterior(alpha, alpha_star, beta, beta_star, gamma, 
    gamma_star, post):
    """
    Given filtered and smoothed probabilities from the forward and backward
    passes, calculate the posterior marginal of each state.
    The arrays passed in are all logs of their respective probabilities.
    """
    T, M = alpha.shape

    for t in xrange(T):
        norm = -np.inf 
        norm_star = -np.inf

        # gamma = alpha * beta
        for m in xrange(M):
            if t == 0:
                gamma[t, m] = -np.inf
                norm = 0.0
            else:
                gamma[t, m] = alpha[t, m] + beta[t, m]
                norm = np.logaddexp(norm, gamma[t, m])

            gamma_star[t, m] = alpha_star[t, m] + beta_star[t, m]
            norm_star = np.logaddexp(norm_star, gamma_star[t, m])

        # normalize
        for m in xrange(M):
            gamma[t, m] += -norm
            gamma_star[t, m] += -norm_star

        # calculate posterior
        for m in xrange(M):
            if t == 0:
                post[t, m] = 0.0
            else:
                post[t, m] = post[t - 1, m]
                post[t, m] += np.exp(gamma_star[t - 1, m]) - np.exp(gamma[t - 1, m])







