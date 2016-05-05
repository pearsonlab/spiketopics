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

def transform_inputs(eta_cov_chol, log_sig_a, Sig_b_chol, Sig_c_chol, log_A_post, log_pi_post, log_mu_eps, log_ups_eps, log_eta_eps):
    """
    Transform unconstrained variables used by gradient to constrained variables
    used by L.

    Note that because we use covs_from_factors, the inputs are not enforced
    to be triangular and so can be overparameterized/nonidentifiable.
    """
    eta_cov = covs_from_factors(eta_cov_chol)
    sig_a = np.exp(log_sig_a)
    Sig_b = covs_from_factors(Sig_b_chol)
    Sig_c = covs_from_factors(Sig_c_chol)
    A_post = np.exp(log_A_post)
    pi_post = np.exp(log_pi_post)
    mu_eps = np.exp(log_mu_eps)
    ups_eps = np.exp(log_ups_eps)
    eta_eps = np.exp(log_eta_eps - 1)

    return eta_cov, sig_a, Sig_b, Sig_c, A_post, pi_post, mu_eps, ups_eps, eta_eps

def get_xi(parvec, X, dimlist):
    """
    Use current parameter vector (parvec) to update current best estimate of
    xi, since this is not itself a tunable parameter, but a function
    of tunable parameters that needs to be known in advance in order to
    "bootstrap" the computation of L.
    """
    pars = unpack(parvec, dimlist)
    psi, A_tilde, pi_tilde = pars[8:11]

    return HMM_inference(psi, A_tilde, pi_tilde)[0]

def stack_last(arglist):
    """
    Stack arguments along (new) last dimension
    """
    nd = arglist[0].ndim
    return np.transpose(np.array(arglist), tuple(range(1, nd + 1)) + (0,))

def L(N, X, m_a, s_a, m_b, S_b, m_c, S_c, A_prior, pi_prior, h_eps, m_eps,
    v_eps, mu_eta, Sig_eta, mu_a, sig_a, mu_b, Sig_b, mu_c, Sig_c, psi,
    A_tilde, pi_tilde, A_post, pi_post, mu_eps, ups_eps, eta_eps):
    """
    Evidence lower bound for model:

    p-model (generative):
        N_{tu} ~ Poisson(eta_{tu})
        eta_{tu} ~ MvNormal(a_u + sum_r x_{tr} b_{ru} + sum_k z_{tk} c_{ku}, Sig_eps)
        a_u ~ Normal(m_a, s_a)
        b_{.u} ~ MvNormal(m_b, S_b)
        c_{.u} ~ MvNormal(m_c, S_c)
        x_{tr} ~ Given
        z_{.k} ~ MarkovChain(A_k, pi_k)
        A_k ~ MarkovMatrix(A_prior) (i.e., Dirichlet on cols)
        pi_k ~ Dirichlet(pi_prior)
        Sig_eps = diag(sig_eps) * Omega * diag(sig_eps)
        sig_eps = 1 / sqrt(tau_eps)
        tau_eps ~ Gamma(m_eps, v_eps)  (mean, variance parameterization)
        Omega ~ LKJ(h_eps)

    q-model (posterior/recognition):
        eta_{t.} ~ MvNormal(mu_eta_{t}, Sig_eta_{t})
        a_u ~ Normal(mu_a_{u}, sig_a_{u})
        b_{.u} ~ MvNormal(mu_b_{u}, sig_b_{u})
        c_{.u} ~ MvNormal(mu_c_{u}, sig_c_{u})
        z_{.k} ~ HMM(psi, A_tilde, pi_tilde)
        A_k ~ MarkovMatrix(A_post)
        pi_k ~ Dirichlet(pi_post)
        Sig_eps = diag(sig_eps) * Omega * diag(sig_eps)
        tau_eps ~ Gamma(mu_eps, ups_eps)  (mean, variance parameterization)
        sig_eps^2 = 1/tau_eps
        Omega ~ LKJ(eta_eps)
    """
    # get shape information
    T, U = N.shape
    M, K = pi_prior.shape

    ###### HMM pre-calculations ############
    xi, Xi, logZ = HMM_inference(psi, A_tilde, pi_tilde)

    # observations
    elbo = log_observed_spikes(N, mu_eta, Sig_eta)

    # effective log firing rates
    log_tau = mean_log_gamma(mu_eps, ups_eps)
    elbo = elbo + log_bottleneck_variables(mu_eps, log_tau, mu_eta, Sig_eta, mu_a, sig_a, mu_b, Sig_b, X, mu_c, Sig_c, eta_eps, xi)
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
        elbo = elbo + hmm_entropy(psi[..., k], A_tilde[..., k],
            pi_tilde[..., k], xi[..., k], Xi[..., k], logZ[k])

    # noise
    elbo = elbo + expected_log_gamma(m_eps, v_eps, mu_eps, ups_eps)
    elbo = elbo + gamma_entropy(mu_eps, ups_eps)
    elbo = elbo + expected_log_LKJ(h_eps, eta_eps, U)
    elbo = elbo + LKJ_entropy(eta_eps, U)

    return elbo

def HMM_inference(psi, A_tilde, pi_tilde):
    K = psi.shape[-1]

    # psi to get forward-backward params
    pobs = np.exp(psi)
    A = np.exp(A_tilde)
    pi = np.exp(pi_tilde)

    # run forward-backward on each chain
    xi_l = []
    Xi_l = []
    logZ_l = []
    for k in range(K):
        xi_this, logZ_this, Xi_this = fb_infer(A[..., k], pi[..., k], pobs[..., k])
        xi_l.append(xi_this)
        Xi_l.append(Xi_this)
        logZ_l.append(logZ_this)

    xi = stack_last(xi_l)
    Xi = stack_last(Xi_l)
    logZ = np.array(logZ_l)

    return xi, Xi, logZ

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
        out = out - np.exp(mu + 0.5 * np.diagonal(Sig, axis1=-1, axis2=-2))
    else:
        out = out - np.exp(mu + 0.5 * Sig)

    return np.sum(out)

def log_bottleneck_variables(tau, log_tau, mu_eta, Sig_eta, mu_a, sig_a, mu_b, Sig_b, X,
    mu_c, Sig_c, eta_eps, xi):
    """
    E[log p(eta)] where
    eta_{.t} ~ MvNormal(m, S)
    m_{ut} = a_u + \sum_r b_{ru} x_{tr} + \sum_k c_{ku} z_{tk}
    """
    T, R = X.shape
    _, U = mu_eta.shape
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

    # logdet Omega terms
    out = out + T * U * (U - 1) * np.log(2)
    out = out - T * np.sum(log_tau)
    out = out + 2 * T * Elogdet_LKJ(eta_eps, U)

    return -0.5 * out

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
    out = -0.5 * np.einsum('uij,ij->u', Sig, Lam)
    out = out - 0.5 * np.einsum('ij,ui,uj->u', Lam, x, x)
    return np.sum(out)

def mvnormal_entropy(mu, Sig):
    """
    Entropy of multivariate normal.
    """
    out = mu.size * 0.5 * (np.log(np.pi) + 1)

    _, logdet = np.linalg.slogdet(Sig)
    out = out + 0.5 * np.sum(logdet)

    return out

def hmm_entropy(psi, A_tilde, pi_tilde, xi, Xi, logZ):
    """
    Entropy of Hidden Markov Model with parameters
    psi: (T, M) -- log evidence
    A_tilde: (M, M) -- log Markov transition matrix
    pi_tilde: (M,) -- log initial state probability
    xi: (T, M) -- q(z_t)  (posterior marginal)
    Xi: (T-1, M, M) -- q(z_{t + 1}, z_t)  (two-slice marginal)
    logZ: log partition function
    xi, Xi, and logZ come from calling forward-backward on the first
    three arguments
    """
    emission_piece = np.sum(xi * psi)
    initial_piece = np.dot(xi[0], pi_tilde)
    transition_piece = np.sum(Xi * A_tilde)
    logq = emission_piece + initial_piece + transition_piece
    if logq > logZ:
        print("Warning, H = {}".format(-logq + logZ))
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

def expected_log_inverse_gamma(a, b, alpha, beta):
    """
    E[log p(x)] where
    p(x) = Inverse-Gamma(a, b)
    q(x) = Inverse-Gamma(alpha, beta)
    """
    elp = -(a + 1) * (np.log(beta) - digamma(alpha)) - b * (beta / (alpha - 1))
    return np.sum(elp)

def inverse_gamma_entropy(alpha, beta):
    """
    Entropy of Inverse-Gamma distribution
    """
    H = alpha + np.log(beta) + gammaln(alpha) - (1 + alpha) * digamma(alpha)
    return np.sum(H)

def expected_log_gamma(m, v, mu, ups):
    """
    E[log p(x)] where
    p(x) = Gamma(a, b)
    q(x) = Gamma(alpha, beta)
    E[x] = m = a/b
    var[x] = v = a/b**2
    => a = m**2/v, b = m/v
    """
    a = m**2/ v
    b = m / v
    alpha = mu**2 / ups
    beta = mu / ups
    elp = (a - 1) * (digamma(alpha) - np.log(beta)) - b * (alpha / beta)
    return np.sum(elp)

def gamma_entropy(mu, ups):
    """
    Entropy of Inverse-Gamma distribution
    E[x] = mu = a/b
    var[x] = ups = a/b**2
    => alpha = mu**2/ups, beta = mu/ups
    """
    alpha = mu**2 / ups
    beta = mu / ups
    H = alpha - np.log(beta) + gammaln(alpha) + (1 - alpha) * digamma(alpha)
    return np.sum(H)

def mean_log_gamma(m, v):
    """
    E[log x] where
    x ~ Gamma(a, b)
    E[x] = m = a/b
    var[x] = v = a/b**2
    => a = m**2/v, b = m/v
    """
    a = m**2/v
    b = m/v
    return digamma(a) - np.log(b)

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

def Elogdet_LKJ(eta, d):
    """
    E[log |A|] where
    A ~ LKJ(eta) of dimension d
    """
    beta = LKJ_to_beta_pars(eta, d)
    return np.sum(digamma(beta) - digamma(2 * beta))

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

def U_to_vec(U, k=1):
    """
    Convert an upper triangular matrix to a vector.
    **Does not** include diagonal.
    Returned vector is read out from U by rows.
    Assumes matrix is **first two** dimensions of passed array.
    """
    r, c = np.triu_indices(U.shape[0], k)
    return U[r, c, ...]

def vec_to_U(v, k=1):
    """
    Convert a vector to an upper triangular matrix.
    Vector **does not** include diagonal entries.
    Matrix is filled in by rows.
    """
    # get matrix size
    N = v.size
    d = int((2 * k - 1 + np.sqrt((2 * k - 1)**2 + 8 * N))/2)
    U = np.zeros((d, d))
    U[np.triu_indices(d, k)] = v
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

def covs_from_factors(F):
    """
    Turn an array of factor matrices (possibly triangular) into an array of
    covariances. Matrix indices are assumed to be the *last two*.
    That is, we want
    F[..., :, :] * F[..., :, :].T
    """
    return np.einsum('...ij,...kj->...ik', F, F)
