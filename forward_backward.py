"""
Contains code to run the forward-backward algorithm for inference in 
Hidden Markov Models.
"""
from __future__ import division
import numpy as np

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

    # initialize
    a = psi[0] * pi
    alpha[0] = a / np.sum(a)
    logZ[0] = np.log(np.sum(a))
    beta[-1, :] = 1
    beta[-1, :] = beta[-1, :] / np.sum(beta[-1, :])
    
    # forwards
    for t in xrange(1, T):
        a = psi[t] * (A.dot(alpha[t - 1]))
        alpha[t] = a / np.sum(a)
        logZ[t] = np.log(np.sum(a))
        
    # backwards
    for t in xrange(T - 1, 0, -1):
        b = A.T.dot(beta[t] * psi[t])
        beta[t - 1] = b / np.sum(b)
        
    # posterior
    gamma = alpha * beta
    gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
    
    # calculate 2-slice marginal
    Xi = ((beta[1:] * psi[1:])[..., np.newaxis] * alpha[:(T - 1), np.newaxis, :]) * A[np.newaxis, ...]

    #normalize
    Xi = Xi / np.sum(Xi, axis=(1, 2), keepdims=True)

    if np.any(np.isnan(gamma)):
        raise ValueError('NaNs appear in posterior')

    return gamma, np.sum(logZ), Xi

