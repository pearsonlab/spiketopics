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

def L(N, m_a, s_a, m_b, S_b, m_c, S_c, A_prior, pi_prior, h_eps, a_eps, b_eps,
    mu_eta, Sig_eta, mu_a, sig_a, mu_b, Sig_b, mu_c, Sig_c, A_post, pi_post,
    alpha_eps, beta_eps, eta_eps):
    """
    Evidence lower bound for model:

    p-model (generative):
        N_{tu} ~ Poisson(eta_{tu})
        eta_{tu} ~ MvNormal(a_u + sum_r x_{tr} b_{ru} + sum_k z_{tk} c_{ku}, Sig_eps)
        a_u ~ Normal(m_a, s_a)
        b_{.u} ~ MvNormal(m_b, S_b)
        c_{.u} ~ MvNormal(m_c, S_c)
        x_{tr} ~ Given
        z_{.k} ~ HMM(A_k, pi_k)
        A_k ~ MarkovMatrix(A_prior) (i.e., Dirichlet on cols)
        pi_k ~ Dirichlet(pi_prior)
        Sig_eps = diag(sig_eps) * Omega * diag(sig_eps)
        sig_eps^2 ~ Inverse-Gamma(a_eps, b_eps)
        Omega ~ LKJ(h_eps)

    q-model (posterior/recognition):
        eta_{t.} ~ MvNormal(mu_eta_{t}, Sig_eta_{t})
        a_u ~ Normal(mu_a_{u}, sig_a_{u})
        b_{.u} ~ MvNormal(mu_b_{u}, sig_b_{u})
        c_{.u} ~ MvNormal(mu_c_{u}, sig_c_{u})
        z_{.k} ~ HMM(A_post, pi_post)
        Sig_eps = diag(sig_eps) * Omega * diag(sig_eps)
        sig_eps^2 ~ Inverse-Gamma(alpha_eps, beta_eps)
        Omega ~ LKJ(eta_eps)
    """
    # first do preliminary calculations
    tau = alpha_eps / beta_eps  # precision noise scale (tau ~ Ga(alpha, beta))
    log_psi = log_emission_probs(tau, mu_eta, mu_c, Sig_c, xi)
    xi, logZ, Xi = fb_infer(A_post, pi_post, np.exp(log_psi))  # forward-backward

    elbo = log_observed_spikes(N, eta_mean, eta_cov)
    elbo += mvnormal_entropy(eta_mean, eta_cov)
    # elbo += expected_log_normal(m_a, s_a, mu_a, sig_a)
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

def log_emission_probs(tau, mu_eta, mu_c, Sig_c, xi):
    """
    Calculate log psi, where psi \propto p(obs|z)
    """
    xi1 = xi[:, 1, :]
    T, U = mu_eta.shape
    K, _ = mu_c.shape
    lpsi = np.einsum('u,tu,ku->tk', tau, mu_eta, mu_c)
    lpsi += 0.5 * np.einsum('u,ku,ju,tj->tk', tau, mu_c, mu_c, xi1)
    lpsi += 0.5 * np.einsum('u,kj,tj->tk', tau, Sig_c, xi1)
    lpsi += 0.5 * np.einsum('u,ku,ku,tk->tk', tau, mu_c, mu_c, 1 - xi1)
    lpsi += 0.5 * np.einsum('u,kk,tk->tk', tau, Sig_c, 1 - xi1)

    log_psi = np.zeros((T, 2, K))
    log_psi[:, 1, :] = lpsi

    return log_psi

def expected_log_normal(m, s, mu, sig):
    """
    E[log p(x)] where
    p(x) = Normal(m, s)  (s is variance)
    q(x) = Normal(mu, sig)
    """
    out = -0.5 * (sig / s + (mu - m)**2/s)
    return np.sum(out)

def normal_entropy(mu, sig):
    """
    Entropy of multivariate normal.
    sig is **variance**
    """
    out = mu.size * 0.5 * (np.log(np.pi) + 1)

    out += 0.5 * np.sum(np.log(sig))

    return out

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
    *Last* index of alpha is the index of the individual distributions.
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
    *Last* index of alpha is the index of the individual distributions.
    """
    Elog_x = digamma(alpha) - digamma(np.sum(alpha, axis=-1, keepdims=True))
    elp = np.sum((a - 1) * Elog_x, axis=-1)
    elp += -_logB(a)

    return np.sum(elp)

def expected_log_markov(a, alpha):
    """
    Markov matrix is first two axes, with axis 0 the Dirichlet.
    E[log p(x)] where
    p(x_{ij}) = Dirichlet(a_{.j})_i  (i.e., each column (axis=0) a Dirichlet)
    q(x_{ij}) = Dirichlet(alpha_{.j})_i
    """
    # change to convention of Dirichlet, where *last* index of a/alpha is the
    # one for the simplex
    return expected_log_dirichlet(a.T, alpha.T)

def markov_entropy(alpha):
    """
    Entropy of Markov matrix.
    Each column in alpha (axis=0) a Dirichlet
    """
    return dirichlet_entropy(alpha.T)

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

def expected_log_beta(a, b, alpha, beta):
    """
    E[log p(x)] where
    p(x) = Beta(a, b)
    q(x) = Beta(alpha, beta)
    """
    # stack parameters along last axis (our Dirichlet convention)
    # and treat as dirichlet
    new_a = np.stack([a, b], axis=-1)
    new_alpha = np.stack([alpha, beta], axis=-1)
    return expected_log_dirichlet(new_a, new_alpha)

def beta_entropy(alpha, beta):
    """
    Entropy of Beta distribution
    """
    # stack parameters along last axis (our Dirichlet convention)
    # and treat as dirichlet
    new_alpha = np.stack([alpha, beta], axis=-1)
    return dirichlet_entropy(new_alpha)

def expected_log_LKJ(h, eta, d):
    """
    E[log p(x)] where
    p(x) = LKJ(h, d)  # d = matrix dimension
    q(x) = LKJ(eta, d)
    """
    b = LKJ_to_beta_pars(h, d)
    beta = LKJ_to_beta_pars(eta, d)
    return expected_log_beta(b, b, beta, beta)

def LKJ_entropy(eta, d):
    beta = LKJ_to_beta_pars(eta, d)
    return beta_entropy(beta, beta)

def LKJ_to_beta_pars(eta, d):
    """
    Transform LKJ distribution with parameter eta for matrix of dimension d
    to vector of beta distribution parameters:
    p_{i >= 1, j>i; 1...i-1} ~ Beta(b_i, b_i)
    b_i = eta + (d - 1 - i)/2
    """
    idxmat = np.broadcast_to((d - 1 - np.arange(d))/2., (d, d)).T
    bmat =  eta + idxmat
    return U_to_vec(bmat)  # only upper triangle, flattened

def draw_LKJ(eta, d):
    """
    Random draw from the LKJ distribution with parameter eta and dimension d.
    """
    betas = LKJ_to_beta_pars(eta, d)
    cpc = 2 * npr.beta(betas, betas) - 1  # rescale to (-1, 1)
    return corrmat_from_vec(corr_from_cpc(cpc))

def U_to_vec(U):
    """
    Convert an upper triangular matrix to a vector.
    **Does not** include diagonal.
    Returned vector is read out from U by rows.
    Assumes matrix is **first two** dimensions of passed array.
    """
    r, c = np.triu_indices(U.shape[0], k=1)
    return U[r, c, ...]

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

def corrmat_from_vec(v):
    """
    Convert a vector of correlations to a matrix.
    Elements of v are read out row-wise.
    """
    C = vec_to_U(v)
    C = C + C.T  # symmetrize
    np.fill_diagonal(C, 1)
    return C

def corr_from_cpc(v):
    """
    Given a vector of canonical partial correlations (taken from the
    upper triangle by rows), return a vector of correlations in the
    same format.
    Makes use of the relation between partial correlations
    r_{ij;L} = \sqrt{(1 - r_{ik;L}^2)(1 - r_{jk;L}^2)} r_{ij;kL} + r_{ik;L} r_{jk;L}
    """
    U = vec_to_U(v)  # upper triangular matrix of canonical partial correlations
    d = np.shape(U)[0]  # dimension of U
    UU = np.zeros_like(U)  # output matrix
    UU[0,1:] = U[0, 1:]  # already full correlations
    for r in range(1, d):
        for c in range(r + 1, d):
            rho = U[r, c]
            for l in range(r - 1, -1, -1):
                rho = rho * np.sqrt((1 - U[l, c]**2) * (1 - U[l, r]**2)) + U[l, c] * U[l, r]
            UU[r, c] = rho

    return U_to_vec(UU)
