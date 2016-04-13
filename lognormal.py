"""
Lognormal model defined as evidence lower bound. For use with autograd package
and gradient ascent.
"""
from __future__ import division
from __future__ import print_function
from autograd.scipy.special import digamma, gammaln
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from fbi import fb_infer

def pack(*args):
    """
    Pack ndarray objects in args with dimensions given in dimlist into a single vector
    return vector, dimlist (list of dimension tuples)
    """
    x = np.concatenate([np.array(a).reshape(-1) for a in args])
    dimlist = [np.array(a).shape for a in args]
    return x, dimlist

def unpack(x, dimlist):
    """
    Given a list of dimensions of arrays and a flat vector x, return
    a list of arrays of the given dimension extracted from x
    """
    offset = 0
    varlist = []
    for d in dimlist:
        sz = np.prod(d)
        v = x[offset:offset + sz].reshape(d)
        varlist.append(v)
        offset = offset + sz

    return varlist

def stack_last(arglist):
    """
    Stack arguments along (new) last dimension
    """
    nd = arglist[0].ndim
    return np.transpose(np.array(arglist), tuple(range(1, nd + 1)) + (0,))

def L(N, X, m_a, s_a, m_b, S_b, m_c, S_c, A_prior, pi_prior, h_eps, a_eps,
    b_eps, mu_eta, Sig_eta, mu_a, sig_a, mu_b, Sig_b, mu_c, Sig_c, A_post,
    pi_post, alpha_eps, beta_eps, eta_eps, xi0):
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
    # get shape information
    T, U = N.shape
    M, K = pi_prior.shape

    ###### noise pre-calculations ############
    tau = alpha_eps / beta_eps  # noise precisions (tau ~ Ga(alpha, beta))

    ###### HMM pre-calculations ############
    log_psi, xi, Xi, logZ = HMM_inference(xi0, tau, mu_eta, mu_c, Sig_c, A_post, pi_post)

    # observations
    elbo = log_observed_spikes(N, mu_eta, Sig_eta)

    # effective log firing rates
    elbo = elbo + log_bottleneck_variables(tau, mu_eta, Sig_eta, mu_a, sig_a, mu_b, Sig_b, X, mu_c, Sig_c, xi)
    elbo = elbo + mvnormal_entropy(mu_eta, Sig_eta)

    # baselines
    elbo = elbo + expected_log_normal(m_a, s_a, mu_a, sig_a)
    elbo = elbo + normal_entropy(mu_a, sig_a)

    # external regressor series coefficients
    elbo = elbo + expected_log_mvnormal(m_b, S_b, mu_b, Sig_b)
    elbo = elbo + mvnormal_entropy(mu_b, Sig_b)

    # HMM coefficients
    elbo = elbo + expected_log_mvnormal(m_c, S_c, mu_c, Sig_c)
    elbo = elbo + mvnormal_entropy(mu_c, Sig_c)

    # HMM parameters
    elbo = elbo + expected_log_markov(A_prior, A_post)
    elbo = elbo + markov_entropy(A_post)
    elbo = elbo + expected_log_dirichlet(pi_prior, pi_post)
    elbo = elbo + dirichlet_entropy(pi_post)

    # HMMs
    Elog_A = mean_log_markov(A_post)
    Elog_pi = mean_log_dirichlet(pi_post)
    for k in range(K):
        elbo = elbo + expected_log_state_sequence(xi[..., k], Xi[..., k],
            Elog_A[..., k], Elog_pi[..., k])
        elbo = elbo + hmm_entropy(log_psi[..., k], np.log(A_post[..., k]),
            np.log(pi_post[..., k]), xi[..., k], Xi[..., k], logZ[k])

    # noise
    elbo = elbo + expected_log_inverse_gamma(a_eps, b_eps, alpha_eps, beta_eps)
    elbo = elbo + inverse_gamma_entropy(alpha_eps, beta_eps)
    elbo = elbo + expected_log_LKJ(h_eps, eta_eps, U)
    elbo = elbo + LKJ_entropy(eta_eps, U)

    return elbo

def HMM_inference(xi0, tau, mu_eta, mu_c, Sig_c, A_post, pi_post):
    K = xi0.shape[-1]

    # "effective" A and pi for the HMM are exp(mean(log(.)))
    Elog_A = mean_log_markov(A_post)
    Elog_pi = mean_log_dirichlet(pi_post)
    A_bar = np.exp(Elog_A)
    pi_bar = np.exp(Elog_pi)

    # first, get emission probabilities
    log_psi = log_emission_probs(tau, mu_eta, mu_c, Sig_c, xi0)

    # run forward-backward on each chain
    xi_l = []
    Xi_l = []
    logZ_l = []
    for k in range(K):
        xi_this, logZ_this, Xi_this = fb_infer(A_bar[..., k], pi_bar[..., k], np.exp(log_psi[..., k]))
        xi_l.append(xi_this)
        Xi_l.append(Xi_this)
        logZ_l.append(logZ_this)

    xi = stack_last(xi_l)
    Xi = stack_last(Xi_l)
    logZ = np.array(logZ_l)

    return log_psi, xi, Xi, logZ

def log_observed_spikes(N, mu, Sig):
    """
    log probability of observed spikes given eta
    N: (T, U)
    mu: (T, U)
    Sig: (T, U, U) or (T, U)
    """
    out = N * mu
    if len(Sig.shape) == 3:
        # get diagonal of *last* two indices
        out = out - np.exp(mu + 0.5 * np.diagonal(Sig, 0, -1, -2))
    else:
        out = out - np.exp(mu + 0.5 * Sig)

    return np.sum(out)

def log_bottleneck_variables(tau, mu_eta, Sig_eta, mu_a, sig_a, mu_b, Sig_b, X,
    mu_c, Sig_c, xi):
    """
    E[log p(eta)] where
    eta_{.t} ~ MvNormal(m, S)
    m_{ut} = a_u + \sum_r b_{ru} x_{tr} + \sum_k c_{ku} z_{tk}
    """
    T, R = X.shape
    xi1 = xi[:, 1, :]

    diag_Sig_eta = np.diagonal(Sig_eta, axis1=-1, axis2=-2)
    out = np.sum(np.einsum('u,tu->t', tau, diag_Sig_eta))
    out = out + np.sum(tau * (mu_eta - mu_a - np.dot(X, mu_b.T) - np.dot(xi1, mu_c.T))**2)
    out = out + T * np.sum(tau * sig_a)
    out = out + np.einsum('u,tr,ts,urs->', tau, X, X, Sig_b)
    out = out + np.einsum('u,tk,tj,ukj->', tau, xi1, xi1, Sig_c)

    v = xi1 * (1 - xi1)
    W = np.diagonal(Sig_c, axis1=-1, axis2=-2) + mu_c**2
    out = out + np.sum(np.einsum('u,tk,uk->t', tau, v, W))

    out = out + np.sum(np.linalg.slogdet(Sig_eta)[1])
    return -0.5 * out


def log_emission_probs(tau, mu_eta, mu_c, Sig_c, xi):
    """
    Calculate log psi, where psi \propto p(obs|z)
    """
    xi1 = xi[:, 1, :]
    T, U = mu_eta.shape
    _, K = mu_c.shape
    lpsi = np.einsum('u,tu,uk->tk', tau, mu_eta, mu_c)
    lpsi = lpsi + 0.5 * np.einsum('u,uk,uj,tj->tk', tau, mu_c, mu_c, xi1)
    lpsi = lpsi + 0.5 * np.einsum('u,ukj,tj->tk', tau, Sig_c, xi1)
    lpsi = lpsi + 0.5 * np.einsum('u,uk,uk,tk->tk', tau, mu_c, mu_c, 1 - xi1)
    diag_Sig_c = np.diagonal(Sig_c, axis1=-1, axis2=-2)
    lpsi = lpsi + 0.5 * np.einsum('u,uk,tk->tk', tau, diag_Sig_c, 1 - xi1)

    # this is a dirty, dirty hack to make autograd work:
    # 1. We can't do the reasonable thing -- assigning to an array of 0s
    #   for z = 1 slice -- because autograd doesn't handle slice assignment
    # 2. **However**, only *relative* psi is important, so we make
    #   log p(D|z=0) = -log p(D|z=1) such that their difference is correct
    log_psi = 0.5 * np.transpose(np.array([-lpsi, lpsi]), (1, 0, 2))

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

    out = out + 0.5 * np.sum(np.log(sig))

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
    out = -0.5 * np.einsum('...ij,...ij->...', Sig, Lam)
    out = out - 0.5 * np.einsum('...ij,...i,...j->...', Lam, x, x)
    return np.sum(out)

def mvnormal_entropy(mu, Sig):
    """
    Entropy of multivariate normal.
    """
    out = mu.size * 0.5 * (np.log(np.pi) + 1)

    _, logdet = np.linalg.slogdet(Sig)
    out = out + 0.5 * np.sum(logdet)

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
    initial_piece = np.dot(xi[0], log_pi)
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
    initial_piece = np.dot(xi[0], Elog_pi)
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
    H = H + (alpha0 - alpha.shape[-1]) * digamma(alpha0)
    H = H - np.sum((alpha - 1) * digamma(alpha), axis=-1)

    return np.sum(H)

def expected_log_dirichlet(a, alpha):
    """
    E[log p(x)] where
    p(x) = Dirichlet(a)
    q(x) = Normal(alpha)
    *Last* index of alpha is the index of the individual distributions.
    """
    Elog_x = mean_log_dirichlet(alpha)
    elp = np.sum((a - 1) * Elog_x, axis=-1)
    elp = elp - _logB(a)

    return np.sum(elp)

def mean_log_dirichlet(alpha):
    """
    E[log x] where
    x ~ Dirichlet(alpha)
    """
    return digamma(alpha) - digamma(np.sum(alpha, axis=-1, keepdims=True))

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

def mean_log_markov(alpha):
    """
    E[log x] where
    x ~ Markov matrix
    """
    return mean_log_dirichlet(alpha.T).T

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
    new_a = stack_last([a, b])
    new_alpha = stack_last([alpha, beta])
    return expected_log_dirichlet(new_a, new_alpha)

def beta_entropy(alpha, beta):
    """
    Entropy of Beta distribution
    """
    # stack parameters along last axis (our Dirichlet convention)
    # and treat as dirichlet
    new_alpha = stack_last([alpha, beta])
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
