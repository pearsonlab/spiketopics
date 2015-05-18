"""
Contains code to run the forward-backward algorithm for inference in 
Hidden Semi-Markov Models.
"""
from __future__ import division
import numpy as np
from numba import jit, autojit

def fb_infer(logA, logpi, logpsi, durations, logpd):
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
    durations is an M x D vector of possible duration values for the 
        hidden states
    logpd is an M x D vector of log probabilities for these durations
    """
    if np.any(logA > 0):
        raise ValueError('Transition matrix probabilities > 1')
    if np.any(logpi > 0):
        raise ValueError('Initial state probabilities > 1')

    # get shapes, preallocate arrays
    T = logpsi.shape[0]
    M, D = logpd.shape

    # normalize p(d)
    lpd = logpd #- np.logaddexp.reduce(logpd, 1, keepdims=True)

    # make sure durations are integer
    dvec = durations.astype('int64')

    alpha = np.empty((T + 1, M))
    alpha_star = np.empty((T + 1, M))
    beta = np.empty((T + 1, M))
    beta_star = np.empty((T + 1, M))

    B = np.empty((T + 1, M, D))
    cum_log_psi = np.empty((T + 1, M))
    _calc_B(dvec, logpsi, B, cum_log_psi)
    del cum_log_psi  # free some memory

    # forward pass
    _forward(alpha, alpha_star, logA, logpi, B, dvec, lpd)

    # backward pass
    _backward(beta, beta_star, logA, B, dvec, lpd) 

    # calculate normalization constant
    logZ = _calc_logZ(alpha)

    # calculate posterior
    gamma = np.empty((T + 1, M))
    gamma_star = np.empty((T + 1, M))
    post = np.empty((T + 1, M))
    _calc_posterior(alpha, alpha_star, beta, beta_star, gamma, 
        gamma_star, logZ, post)
    del gamma, gamma_star

    # calculate two-slice marginals
    logXi = np.empty((T - 1, M, M))
    _calc_two_slice(alpha, beta_star, logA, logXi)

    # calculate log of max-likelihood estimates of p(d|z)
    logC = np.empty((M, D))
    _estimate_duration_dist(alpha_star, beta, B, dvec, lpd, logZ, logC)

    return post[1:], logZ, np.exp(logXi), logC

@jit("void(int64[:], float64[:, :], float64[:, :, :], float64[:, :])", 
    nopython=True)
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
                prev = max(0, t - d)
                B[t, m, ix] = cum_log_psi[t, m] - cum_log_psi[prev, m]

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
            a[t, m] = -np.inf
            for ix, d in enumerate(dvec):
                if t >= d:
                    a[t, m] = np.logaddexp(B[t, m, ix] + D[m, ix] + 
                        astar[t - d, m], a[t, m])
                else:
                    break

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

    # initialize (remember, it's on log scale)
    for m in xrange(M):
        b[T - 1, m] = 0
        bstar[T - 1, m] = 0

    for t in xrange(T - 2, -1, -1):
        for m in xrange(M):
            # calculate b^*[t, m]
            bstar[t, m] = -np.inf
            for ix, d in enumerate(dvec):
                if t + d < T:
                    bstar[t, m] = np.logaddexp(b[t + d, m] + 
                        B[t + d, m, ix] + D[m, ix], bstar[t, m])

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

@jit("void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64, float64[:, :])", nopython=True)
def _calc_posterior(alpha, alpha_star, beta, beta_star, gamma, 
    gamma_star, logZ, post):
    """
    Given filtered and smoothed probabilities from the forward and backward
    passes, calculate the posterior marginal of each state.
    The arrays passed in are all logs of their respective probabilities.

    NOTE: gamma and gamma_star should NOT need to be normalized separately, 
    but reduce operations with logaddexp are not as stable as necessary.
    """
    T, M = alpha.shape

    for t in xrange(T):
        # gamma = alpha * beta
        for m in xrange(M):
            if t == 0:
                gamma[t, m] = -np.inf
            else:
                gamma[t, m] = alpha[t, m] + beta[t, m] - logZ

            gamma_star[t, m] = alpha_star[t, m] + beta_star[t, m] - logZ

        # calculate posterior
        for m in xrange(M):
            if t == 0:
                post[t, m] = 0.0
            else:
                post[t, m] = post[t - 1, m]
                post[t, m] += np.exp(gamma_star[t - 1, m]) - np.exp(gamma[t - 1, m])

@jit("void(float64[:, :], float64[:, :], float64[:, :], float64[:, :, :])", nopython=True)
def _calc_two_slice(alpha, beta_star, A, Xi):
    """
    Calculate two-slice smoothed marginals, E[z_{t + 1} z_{t}].
    As usual, all quantities are logged.
    """
    T, M = alpha[1:, :].shape
    a = alpha[1:, :]
    bstar = beta_star[1:, :]

    for t in xrange(T - 1):
        norm = -np.inf
        for i in xrange(M):
            for j in xrange(M):
                Xi[t, i, j] = bstar[t, i] + A[i, j] + a[t, j]
                norm = np.logaddexp(norm, Xi[t, i, j])

        # normalize joint distribution
        for i in xrange(M):
            for j in xrange(M):
                Xi[t, i, j] += -norm

@jit("void(float64[:, :], float64[:, :], float64[:, :, :], int64[:], float64[:, :], float64, float64[:, :])", nopython=True)
def _estimate_duration_dist(a_star, b, B, dvec, D, logZ, C):
    """
    Calculate sufficient statistics for maximum likelihood estimation
    of p(d|z).
    """
    T, M, _ = B.shape

    for m in xrange(M):
        for ix, d in enumerate(dvec):
            C[m, ix] = -np.inf
            for t in xrange(1, T):
                if t >= d:
                    C[m, ix] = np.logaddexp(C[m, ix], a_star[t - d, m] +
                     D[m, ix] + B[t, m, ix] + b[t, m])
            C[m, ix] += -logZ






