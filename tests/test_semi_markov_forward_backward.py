"""
Tests for Forwards-Backwards inference for semi-Markov model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_true, set_trace
import numpy as np
import scipy.stats as stats
import numpy.testing as npt
import hsmm_forward_backward as fb

def setup_hsmm():
    np.random.rand(12345)

    T, K, D = _setup_constants()

    A, pi = _make_transition_probs(T, K)

    chain, dur_mean, dur_std = _make_chain(A, pi, T, K)

    y, mu, sig = _make_fb_data(chain)

    logpsi = _calc_emission_probs(y, mu, sig)

    dvec, logpd = _make_duration_dist(dur_mean, dur_std, D)

def _setup_constants():
    """
    These variables determine problem size.
    """
    T = 500  # times
    K = 4  # levels of hidden state
    D = 50

    return T, K, D

def _make_transition_probs(T, K):
        """
        Make a Markov transition matrix and initial state vector.
        Columns of A sum to 1, so A acts to the right.
        """ 
        lo, hi = 1, 20
        rows = []
        for _ in xrange(K):
            alpha = stats.randint.rvs(lo, hi, size=K)
            row = stats.dirichlet.rvs(alpha)
            rows.append(row)
        A = np.vstack(rows).T

        alpha = stats.randint.rvs(lo, hi, size=K)
        pi = stats.dirichlet.rvs(alpha).squeeze()

        return A, pi

def _make_chain(A, pi, T, K):
    """
    Make a Markov chain by using the transition probabilities.
    """

    # make the chain by evolving each category through time
    chain = np.empty((K, T), dtype='int')

    # pick pars for duration distribution
    mm = 10 * np.random.rand(K) # mean
    ss = 3 * np.random.rand(K) # standard deviation

    # initialize
    t = 0
    while t < T:
        if t == 0:
            pp = pi
        else:
            pp = A.dot(chain[:, t - 1])

        # pick a new state
        newstate = np.random.multinomial(1, pp)[:, np.newaxis]
        k = np.argmax(newstate)

        # pick a duration
        d = np.rint(stats.norm.rvs(loc=mm[k], scale=ss[k])).astype('int')
        d = np.min([d, T - d])

        # fill in the next d steps of the chain
        chain[:, t:(t+d)] = newstate
        t += d
        
    return chain, mm, ss

def _make_fb_data(chain):
    K, T = chain.shape
    mu = 10 * np.random.rand(K)
    sig = 2 * np.random.rand(K)

    y = stats.norm.rvs(loc=mu.dot(chain), scale=sig.dot(chain), 
        size=T)

    return y, mu, sig

def _calc_emission_probs(y, mu, sig):
    logpsi = stats.norm.logpdf(y[:, np.newaxis], loc=mu[np.newaxis, :], 
        scale=sig[np.newaxis, :])

    return logpsi

def _make_duration_dist(mu, sig, D):
    dvec = np.arange(D)
    logpdf = stats.norm.logpdf(dvec[np.newaxis, :], loc=mu[:, np.newaxis],
        scale=sig[:, np.newaxis])

    # normalize
    logpdf -= np.log(np.sum(np.exp(logpdf), axis=1, keepdims=True))

    return dvec, logpdf

if __name__ == '__main__':
    np.random.rand(12345)

    T, K, D = _setup_constants()

    A, pi = _make_transition_probs(T, K)

    chain, dur_mean, dur_std = _make_chain(A, pi, T, K)

    y, mu, sig = _make_fb_data(chain)

    logpsi = _calc_emission_probs(y, mu, sig)

    dvec, logpd = _make_duration_dist(dur_mean, dur_std, D)
