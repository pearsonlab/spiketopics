"""
Lognormal model defined as evidence lower bound. For use with autograd package
and gradient ascent.
"""
from __future__ import division
from __future__ import print_function
from scipy.special import digamma, gammaln
import autograd.numpy as np
import autograd.numpy.random as npr
from fbi import fb_infer

def L(N, eta_mean, eta_cov):
    elbo = log_observed_spikes(N, eta_mean, eta_cov)
    elbo += normal_entropy(eta_mean, eta_cov)
    return elbo

def log_observed_spikes(N, mu, Sig):
    """
    log probability of observed spikes given eta
    N: (T, U)
    mu: (T, U)
    Sig: (T, U, U) or (T, U)
    """
    out = N * mu
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
    x = m - mu
    out = -0.5 * np.einsum('...ij, ...ij', Sig, Lam)
    out += -0.5 * np.einsum('...ij, ...i, ...j', Lam, x, x)
    return np.sum(out)

def normal_entropy(mu, sig):
    """
    Entropy of multivariate normal.
    sig is **variance**
    """
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
    initial_piece = xi[0].dot(log_pi)
    transition_piece = np.sum(Xi * log_A)
    logq = emission_piece + initial_piece + transition_piece
    return -logq + logZ

def expected_log_state_sequence(xi, Xi, Elog_A, Elog_pi):
    """
    Expected log of Markov state sequence
    xi: (T, M) -- q(z_t)  (posterior marginal)
    Xi: (T-1, M, M) -- q(z_{t + 1}, z_t)  (two-slice marginal)
    Elog_A: (M, M) -- Expected value of log transition matrix
    Elog_pi: (M,) -- Expected value of log initial state probability
    """
    initial_piece = xi[0].dot(Elog_pi)
    transition_piece = np.sum(Xi * Elog_A)
    return initial_piece + transition_piece

def _logB(alpha):
    """
    logB = sum_i log Gamma(alpha_i) - log Gamma(sum_i alpha_i)
    """
    return np.sum(gammaln(alpha), axis=-1) - gammaln(np.sum(alpha, axis=-1))

def dirichlet_entropy(alpha):
    """
    Entropy of collection of Dirichlet distributions.
    *Last* index of alpha is the index that sums to 1.
    """
    alpha0 = np.sum(alpha, axis=-1)
    H = _logB(alpha)
    H += (alpha0 - alpha.shape[-1]) * digamma(alpha0)
    H += -np.sum((alpha - 1) * digamma(alpha), axis=-1)

    return np.sum(H)

def expected_log_dirichlet(a, alpha):
    """
    E[log p(x)] where
    p(x) = Dirichlet(a)
    q(x) = Normal(alpha)
    *Last* index of a/alpha is the index that sums to 1.
    """
    Elog_x = digamma(alpha) - digamma(np.sum(alpha, axis=-1, keepdims=True))
    elp = np.sum((a - 1) * Elog_x, axis=-1)
    elp += -_logB(a)

    return np.sum(elp)

def inverse_gamma_entropy(alpha, beta):
    """
    Entropy of Inverse-Gamma distribution
    """
    H = alpha + np.log(beta) + gammaln(alpha) - (1 + alpha) * digamma(alpha)
    return np.sum(H)

def expected_log_inverse_gamma(a, b, alpha, beta):
    """
    E[log p(x)] where
    p(x) = Inverse-Gamma(a, b)
    q(x) = Inverse-Gamma(alpha, beta)
    """
    elp = -(a + 1) * (np.log(beta) - digamma(alpha)) - b * (beta / (alpha - 1))
    return np.sum(elp)

def U_to_vec(U):
    """
    Convert an upper triangular matrix to a vector.
    **Does not** include diagonal.
    Returned vector is read out from U by rows.
    """
    r, c = np.triu_indices_from(U, k=1)
    return U[r, c]

def vec_to_U(v):
    """
    Convert a vector to an upper triangular matrix.
    Vector **does not** include diagonal entries.
    Matrix is filled in by rows.
    """
    # get matrix size
    N = v.size
    d = int((1 + np.sqrt(1 + 8 * N))/2)
    U = np.zeros((d, d))
    U[np.triu_indices(d, 1)] = v
    return U

def cor_from_vec(v):
    """
    Convert a vector of partial correlations to a matrix.
    Elements of v are read out row-wise.
    """
    C = vec_to_U(v)
    C = C + C.T  # symmetrize
    np.fill_diagonal(C, 1)
    return C
