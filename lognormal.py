"""
Lognormal model defined as evidence lower bound. For use with autograd package
and gradient ascent.
"""
from __future__ import division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr

def L(N, eta_mean, eta_cov):
    elbo = log_observed_spikes(N, eta_mean, eta_cov)
    elbo += log_fr_entropy(eta_mean, eta_cov)
    return elbo

def log_observed_spikes(N, mu, Sig):
    """
    log probability of observed spikes given eta
    N: (T, U)
    mu: (T, U)
    Sig: (T, U, U) or (T, U)
    """
    out = (N + 1) * mu
    if len(Sig.shape) == 3:
        out += - np.exp(mu + 0.5 * np.diagonal(Sig, 0, 1, 2))
    else:
        out += - np.exp(mu + 0.5 * Sig)

    return np.sum(out)

def log_fr_entropy(mu, Sig):
    T, U = mu.shape
    out = T * U * 0.5 * (np.log(np.pi) + 1)

    if len(Sig.shape) == 3:
        # if Sig is (T, U, U), does logdet over last two dims
        _, logdet = np.linalg.slogdet(Sig)
        out += 0.5 * np.sum(logdet)
    else:
        out += 0.5 * np.sum(np.log(Sig))

    return out
