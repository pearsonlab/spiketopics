"""
Contains code to run the forward-backward algorithm for inference in 
Hidden Semi-Markov Models. Based on Yu and Kobayashi (2006)
"""
from __future__ import division
import numpy as np
from numba import jit

def fb_infer(logA, logpi, logpsi, durations, logpd):
    """
    Implement the forward-backward inference algorithm.
    logA is a matrix of log transition probabilities that acts to the right:
        new_state = A * old_state, so that columns of A sum to 1
    logpi is a matrix of log initial state probabilities
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

    # normalize distributions
    # no need to normalize psi; Z naturally scales to compensate
    pi_norm = np.logaddexp.reduce(logpi)
    A_norm = np.logaddexp.reduce(logA, 0, keepdims=True)
    pd_norm = np.logaddexp.reduce(logpd, 1, keepdims=True)
    lpi = logpi - pi_norm
    lA = logA - A_norm
    lpd = logpd - pd_norm
    lpsi = logpsi

    # make sure durations are integer
    dvec = durations.astype('int64')

    logalpha = np.empty((T, M, D)) 
    logbeta = np.empty((T, M, D))
    logZ = np.empty(T,)
    logE = np.empty((T, M))
    logS = np.empty((T, M))
    logEstar = np.empty((T, M))
    logSstar = np.empty((T, M))
    logxi = np.empty((T, M))
    logXi = np.empty((T - 1, M, M))
    logC = np.empty((T, M, D))

    _forward(logalpha, logZ, logE, logS, lpi, lA, lpsi, dvec, lpd)
    _backward(logbeta, logZ, logEstar, logSstar, lA, lpsi, dvec, lpd)
    _posterior(logalpha, logbeta, logxi)
    _two_slice(logE, logEstar, lA, logXi)
    _sequence_entry(logS, lpd, lpi, logbeta, logC)
    logZtot = _logZ_tot(logZ, pi_norm, A_norm, pd_norm, logXi, logC)

    return np.exp(logxi), logZtot, np.exp(logXi), np.exp(logC)

@jit("void(float64[:, :, :], float64[:], float64[:, :], float64[:, :], float64[:], float64[:, :], float64[:, :], int64[:], float64[:, :])", nopython=True)
def _forward(logalpha, logZ, logE, logS, logpi, logA, logpsi, dvec, logpd):
    """
    Perform the forward pass for the forward-backward algorithm.
    """
    T, M, D = logalpha.shape

    for t in xrange(T):
        logZ[t] = -np.inf

        # alpha
        for m in xrange(M):
            for ix, _ in enumerate(dvec):

                if t == 0:
                    logalpha[t, m, ix] = logpi[m] + logpd[m, ix]
                else:
                    trans = logS[t - 1, m] + logpd[m, ix]
                    if ix < D - 1:
                        stay = (logpsi[t - 1, m] - logZ[t - 1] + 
                            logalpha[t - 1, m, ix + 1])
                    else:
                        stay = -np.inf
                    logalpha[t, m, ix] = np.logaddexp(trans, stay)

                # Z
                logZ[t] = np.logaddexp(logZ[t], logalpha[t, m, ix] + 
                    logpsi[t, m])

        # E
        for m in xrange(M):
            logE[t, m] = -np.inf
            logE[t, m] = logalpha[t, m, 0] + logpsi[t, m] - logZ[t]

        # S
        for m in xrange(M):
            logS[t, m] = -np.inf
            for n in xrange(M):
                logS[t, m] = np.logaddexp(logS[t, m], logA[m, n] + 
                    logE[t, n])


@jit("void(float64[:, :, :], float64[:], float64[:, :], float64[:, :],  float64[:, :], float64[:, :], int64[:], float64[:, :])", nopython=True)
def _backward(logbeta, logZ, logEstar, logSstar, logA, logpsi, dvec, logpd):
    """
    Perform the backward pass of the forward-backward algorithm.
    """
    T, M, D = logbeta.shape

    for t in xrange(T - 1, -1, -1):
        for m in xrange(M):
            logEstar[t, m] = -np.inf
            logSstar[t, m] = -np.inf

            for ix, _ in enumerate(dvec):
                # calculate beta
                if t == T - 1:
                    logbeta[t, m, ix] = logpsi[t, m] - logZ[t]
                else:
                    if ix == 0:
                        logbeta[t, m, ix] = (logSstar[t + 1, m] + 
                            logpsi[t, m] - logZ[t])
                    else:
                        logbeta[t, m, ix] = (logbeta[t + 1, m, ix - 1] +
                            logpsi[t, m] - logZ[t])

                # Estar
                logEstar[t, m] = np.logaddexp(logEstar[t, m], 
                    logbeta[t, m, ix] + logpd[m, ix])

        # Sstar
        for m in xrange(M):
            for n in xrange(M):
                logSstar[t, m] = np.logaddexp(logSstar[t, m], 
                    logEstar[t, n] + logA[n, m])

@jit("void(float64[:, :, :], float64[:, :, :], float64[:, :])", nopython=True)
def _posterior(logalpha, logbeta, logxi):
    """
    Calculate posterior probability of being in state m at time t.
    """
    T, M, D = logalpha.shape

    for t in xrange(T):
        for m in xrange(M):
            logxi[t, m] = -np.inf
            for d in xrange(D):
                logxi[t, m] = np.logaddexp(logxi[t, m], logalpha[t, m, d] + 
                    logbeta[t, m, d])

@jit("void(float64[:, :], float64[:, :], float64[:, :], float64[:, :, :])", nopython=True)
def _two_slice(logE, logEstar, logA, logXi):
    """
    Calculate posterior two-slice marginal: probability of transitioning 
    from state (m, t) -> (n, t+1)
    """
    T, M = logE.shape

    for t in xrange(T - 1):
        for n in xrange(M):
            for m in xrange(M):
                logXi[t, n, m] = logEstar[t + 1, n] + logA[n, m] + logE[t, m]

@jit("void(float64[:, :], float64[:, :], float64[:], float64[:, :, :], float64[:, :, :])", nopython=True)
def _sequence_entry(logS, logpd, logpi, logbeta, logC):
    """
    Calculate posterior probability of entering state m for a duration d 
    starting at time t.
    """
    T, M, D = logbeta.shape

    for t in xrange(T):
        for m in xrange(M):
            for d in xrange(D):
                logC[t, m, d] = logbeta[t, m, d] + logpd[m, d]

                if t == 0:
                    logC[t, m, d] += logpi[m]
                else:
                    logC[t, m, d] += logS[t - 1, m]

@jit("float64(float64[:], float64, float64[:, :], float64[:, :], float64[:, :, :], float64[:, :, :])", nopython=True)
def _logZ_tot(logZ, pi_norm, A_norm, pd_norm, logXi, logC):
    """
    Calculate corrections to logZ that derive from normalizing inputs. This
    is needed if we want logZ to correctly cancel unnormalized inputs in 
    log probability of the posterior. Because of this, logZ is no longer
    p(y_{1:T}), but a pure normalization factor.
    """
    T, M, D = logC.shape

    # pi
    logZtot = pi_norm 

    # logZ[t] = p(y_t|y_{1:t - 1})
    for t in xrange(T):
        logZtot += logZ[t]

    # A
    for t in xrange(T - 1):
        for m in xrange(M):
            for n in xrange(M):
                logZtot += np.exp(logXi[t, m, n]) * A_norm[0, n]

    # p(d)
    for t in xrange(T):
        for m in xrange(M):
            for d in xrange(D):
                logZtot += np.exp(logC[t, m, d]) * pd_norm[m, 0]

    return logZtot