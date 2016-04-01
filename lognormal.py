"""
Lognormal model defined as evidence lower bound. For use with autograd package
and gradient ascent.
"""
from __future__ import division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from fbi import fb_infer

def L(N, eta_mean, eta_cov):
    elbo = log_observed_spikes(N, eta_mean, eta_cov)
    elbo += mvnormal_entropy(eta_mean, eta_cov)
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

def expected_log_normal(m, s, mu, sig):
    """
    E[log p(x)] where
    p(x) = Normal(m, s)  (s is variance)
    q(x) = Normal(mu, sig)
    """
    out = -0.5 * (sig / s + (mu - m)**2/s)
    return np.sum(out)

def expected_log_mvnormal(m, S, mu, Sig):
    """
    E[log p(x)] where
    p(x) = Normal(m, S)  (s is variance)
    q(x) = Normal(mu, Sig)
    Vector index of normal is assumed to be *last* index
    """
    Lam = np.linalg.inv(S)
    x = mu - mu
    out = -0.5 * (Sig * Lam + np.einsum('...ij, ...i, ...j', Lam, x, x))
    return np.sum(out)

def normal_entropy(mu, sig):
    """
    Entropy of multivariate normal.
    sig is **variance**
    """
    T, U = mu.shape
    out = mu.size * 0.5 * (np.log(np.pi) + 1)

    out += 0.5 * np.sum(np.log(sig))

    return out

def mvnormal_entropy(mu, Sig):
    """
    Entropy of multivariate normal.
    """
    out = mu.size * 0.5 * (np.log(np.pi) + 1)

    _, logdet = np.linalg.slogdet(Sig)
    out += 0.5 * np.sum(logdet)

    return out

def hmm_entropy(log_psi, log_A, log_pi, xi, Xi, logZ):
    """
    Entropy of Hidden Markov Model with parameters
    log_psi: (T, M) -- log evidence
    log_A: (M, M) -- log Markov transition matrix
    log_pi: (M,) -- log initial state probability
    xi: (T, M) -- q(z_t)  (posterior marginal)
    Xi: (T-1, M, M) -- q(z_{t + 1}, z_t)  (two-slice marginal)
    logZ: log partition function
    xi, Xi, and logZ come from calling forward-backward on the first
    three arguments
    """
    emission_piece = np.sum(xi * log_psi)
    initial_piece = xi[:, 0].dot(log_pi)
    transition_piece = np.sum(Xi * log_A)
    logq = emission_piece + initial_piece + transition_piece
    return -logq + logZ

def expected_log_state_sequence(xi, Xi, EA, Epi):
    """
    Expected log of Markov state sequence
    xi: (T, M) -- q(z_t)  (posterior marginal)
    Xi: (T-1, M, M) -- q(z_{t + 1}, z_t)  (two-slice marginal)
    Elog_A: (M, M) -- Expected value of log transition matrix
    Elog_pi: (M,) -- Expected value of log initial state probability
    """
    initial_piece = xi[:, 0].dot(Elog_pi)
    transition_piece = np.sum(Xi * Elog_A)
    return initial_piece + transition_piece
