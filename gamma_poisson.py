import numpy as np
import scipy.stats as stats

def calculate_observation_probs(y, pars, rate_min=1e-200):
    """
    Calculates the probability of the observations given the hidden state: 
    p(y_t|z_t, pars). Calculates on the log scale to avoid overflow/underflow 
    issues.
    """
    T = y.shape[0]  # number of observations
    M = pars.shape[-1]  # last index is for values of z
    logpsi = np.empty((T, M))

    # sanitize inputs, since logpmf(lam = 0) = nan
    rates = pars.copy()
    rates[rates == 0] = rate_min

    # Poisson observation model
    # observation matrix is times x units
    # z = 0
    logpsi[:, 0] = np.sum(stats.poisson.logpmf(y, rates[..., 0]), axis=1)

    # z = 1
    logpsi[:, 1] = np.sum(stats.poisson.logpmf(y, rates[..., 1]), axis=1)

    # take 
    
    # take care of underflow
    logpsi = logpsi - np.amax(logpsi, 1, keepdims=True)
    psi = np.exp(logpsi)

    if np.any(np.isnan(psi)):
        raise ValueError('NaNs appear in observation probabilities.')

    return psi


def fb_infer(y, lam, A, pi0):
    """
    Implement the forward-backward inference algorithm.
    y is a times x units matrix of observations (counts)
    lam is a times x units x states array of Poisson rates
    A is a matrix of transition probabilities that acts to the right:
    new_state = A * old_state, so that columns of A sum to 1
    """
    T = y.shape[0]
    M = A.shape[0]

    # initialize empty variables
    alpha = np.empty((T, M))  # p(z_t|y_{1:T})
    beta = np.empty((T, M))  # p(y_{t+1:T}|z_t) (unnormalized)
    gamma = np.empty((T, M))  # p(z_t|y_{1:T}) (posterior)
    logZ = np.empty(T)  # log partition function

    psi = calculate_observation_probs(y, lam)
    
    # initialize
    alpha[0, :] = pi0
    beta[-1, :] = 1
    logZ[0] = 0
    
    # forwards
    for t in xrange(1, T):
        a = psi[t, :] * (A.dot(alpha[t - 1, :]))
        logZ[t] = np.sum(a)
        alpha[t, :] = a / np.sum(a)
        
    # backwards
    for t in xrange(T - 1, 0, -1):
        b = A.T.dot(beta[t, :] * psi[t, :])
        beta[t - 1, :] = b / np.sum(b)
        
    # posterior
    gamma = alpha * beta
    gamma = gamma / np.sum(gamma, 1, keepdims=True)
    
    # two-slice marginal matrix: xi = p(z_{t+1}, z_t|y_{1:T})
    beta_shift = np.expand_dims(np.roll(beta * psi, shift=-1, axis=0), 2)

    # take outer product; make sure t axis on alpha < T
    # and t+1 axis on bp > 0
    Xi = beta_shift[1:] * alpha[:-1, np.newaxis, :]

    #normalize
    Xi = Xi / np.sum(Xi, axis=(1, 2), keepdims=True)

    if np.any(np.isnan(gamma)):
        raise ValueError('NaNs appear in posterior')

    return gamma, np.sum(logZ), Xi

class GPModel:
    """
    This class represents and fits a Gamma-Poisson model via variational
    inference. Variables are as follows:
    T: number of (discrete) times
    dt: time difference between observations
    U: number of observation units
    K: number of latent categories to fit

    N: T x U array of count observations
    lam: K x U array of _multiplicative_ effects; lam[0] is baseline
    z: T x K array of latent states in the Markov chain (0 or 1)
    A: 2 x 2 x K array of column-stochastic Markov transition matrices (one
        per chain)
    pi0: 2 x K array of initial chain state probabilities

    Model is defined as:
    N_{tu}|z ~ Poisson(prod_k mu_{ku}^z_{tk})
    lam_{ku}|z ~ Gamma(alpha_{ku}, beta_{ku})
    A(k)_{i -> 1} ~ Beta(gamma1_{ik}, gamma2_{ik})
    pi0(k)_1 ~ Beta(delta1_k, delta2_k)

    Priors are:
    lam_{ku}|z ~ Gamma(c_{ku}, d_{ku})
    A(k)_{i -> 1} ~ Beta(nu1_{ik}, nu2_{ik})
    pi0(k)_1 ~ Beta(rho1_k, rho2_k)
    """
    def __init__(self, T, K, U, dt):
        """
        Set up basic constants for the model. 
        """
        pass

