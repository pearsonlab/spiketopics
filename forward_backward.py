"""
Contains code to run the forward-backward algorithm for inference in 
Hidden Markov Models.
"""
from __future__ import division
import numpy as np
from numba import jit, autojit

def fb_infer(A, pi, psi):
    """
    Implement the forward-backward inference algorithm.
    A is a matrix of transition probabilities that acts to the right:
    new_state = A * old_state, so that columns of A sum to 1
    psi is the vector of evidence: p(y_t|z_t); it does not need to be
    normalized, but the lack of normalization will be reflected in logZ
    such that the end result using the given psi will be properly normalized
    when using the returned value of Z
    """
    if np.any(A > 1):
        raise ValueError('Transition matrix probabilities > 1')
    if np.any(pi > 1):
        raise ValueError('Initial state probabilities > 1')

    T = psi.shape[0]
    M = A.shape[0]

    # initialize empty variables
    alpha = np.empty((T, M))  # p(z_t|y_{1:T})
    beta = np.empty((T, M))  # p(y_{t+1:T}|z_t) (unnormalized)
    gamma = np.empty((T, M))  # p(z_t|y_{1:T}) (posterior)
    logZ = np.empty(T)  # log partition function
    Xi = np.empty((T - 1, M, M))

    # initialize
    a = psi[0] * pi
    alpha[0] = a / np.sum(a)
    logZ[0] = np.log(np.sum(a))
    beta[-1, :] = 1
    beta[-1, :] = beta[-1, :] / np.sum(beta[-1, :])
    
    forward(psi, A, alpha, logZ, a)
        
    backward(psi, A, beta, a)

    # posterior
    calc_post(alpha, beta, gamma)
    
    # calculate 2-slice marginal
    two_slice(alpha, beta, psi, A, Xi)

    if np.any(np.isnan(gamma)):
        raise ValueError('NaNs appear in posterior')

    return gamma, np.sum(logZ), Xi

@autojit(nopython=True)
def forward(psi, A, alpha, logZ, a):
    T = psi.shape[0]
    M = A.shape[0]

    for t in xrange(1, T):
        asum = 0.0
        for i in xrange(M):
            a[i] = 0.0
            for j in xrange(M):
                a[i] += psi[t, i] * A[i, j] * alpha[t - 1, j]
            asum += a[i]

        for i in xrange(M):
            alpha[t, i] = a[i] / asum

        logZ[t] = np.log(asum)

@autojit(nopython=True)
def backward(psi, A, beta, a):
    T = psi.shape[0]
    M = A.shape[0]

    for t in xrange(T - 1, 0, -1):
        asum = 0.0
        for i in xrange(M):
            a[i] = 0.0
            for j in xrange(M):
                a[i] += A[i, j] * beta[t, j] * psi[t, j]
            asum += a[i]

        for i in xrange(M):
            beta[t - 1, i] = a[i] / asum

@autojit(nopython=True)
def calc_post(alpha, beta, gamma):
    T, M = alpha.shape

    for t in xrange(T):
        gamsum = 0.0
        for m in xrange(M):
            gamma[t, m] = alpha[t, m] / beta[t, m]
            gamsum += gamma[t, m]

        for m in xrange(M):
            gamma[t, m] /= gamsum

@autojit(nopython=True)
def two_slice(alpha, beta, psi, A, Xi):
    T, M = alpha.shape

    for t in xrange(T - 1):
        xsum = 0.0
        for i in xrange(M):
            for j in xrange(M):
                Xi[t, i, j] = beta[t + 1, i] * psi[t + 1, i] 
                Xi[t, i, j] *= alpha[t, j] * A[i, j] 
                xsum += Xi[t, i, j]

        # normalize
        for i in xrange(M):
            for j in xrange(M):
                Xi[t, i, j] /= xsum






