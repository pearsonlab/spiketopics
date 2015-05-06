"""
Tests for Forwards-Backwards inference for semi-Markov model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_true, set_trace, assert_raises
import numpy as np
import scipy.stats as stats
import numpy.testing as npt
import hsmm_forward_backward as fb

class Test_Forwards_Backwards:
    @classmethod
    def setup_class(self):
        np.random.rand(12345)

        self._setup_constants()

        self._make_transition_probs()

        self._make_chain()

        self._make_fb_data()

        self._calc_emission_probs()

        self._make_duration_dist()

    @classmethod
    def _setup_constants(self):
        """
        These variables determine problem size.
        """
        self.T = 500  # times
        self.K = 4  # levels of hidden state
        self.D = 50

    @classmethod
    def _make_transition_probs(self):
            """
            Make a Markov transition matrix and initial state vector.
            Columns of A sum to 1, so A acts to the right.
            """ 
            lo, hi = 1, 20
            rows = []
            for _ in xrange(self.K):
                alpha = stats.randint.rvs(lo, hi, size=self.K)
                row = stats.dirichlet.rvs(alpha)
                rows.append(row)
            self.A = np.vstack(rows).T

            alpha = stats.randint.rvs(lo, hi, size=self.K)
            self.pi = stats.dirichlet.rvs(alpha).squeeze()

    @classmethod
    def _make_chain(self):
        """
        Make a Markov chain by using the transition probabilities.
        """

        # make the chain by evolving each category through time
        chain = np.empty((self.K, self.T), dtype='int')

        # pick pars for duration distribution
        mm = 10 * np.random.rand(self.K) # mean
        ss = 3 * np.random.rand(self.K) # standard deviation

        # initialize
        t = 0
        while t < self.T:
            if t == 0:
                pp = self.pi
            else:
                pp = self.A.dot(chain[:, t - 1])

            # pick a new state
            newstate = np.random.multinomial(1, pp)[:, np.newaxis]
            k = np.argmax(newstate)

            # pick a duration
            d = np.rint(stats.norm.rvs(loc=mm[k], scale=ss[k])).astype('int')
            d = np.min([d, self.T - d])

            # fill in the next d steps of the chain
            chain[:, t:(t+d)] = newstate
            t += d
            
        self.chain = chain
        self.dur_mean = mm
        self.dur_std = ss

    @classmethod
    def _make_fb_data(self):
        mu = 10 * np.random.rand(self.K)
        sig = 2 * np.random.rand(self.K)

        y = stats.norm.rvs(loc=mu.dot(self.chain), 
            scale=sig.dot(self.chain), 
            size=self.T)

        self.y = y
        self.obs_mean = mu
        self.obs_std = sig

    @classmethod
    def _calc_emission_probs(self):
        logpsi = stats.norm.logpdf(self.y[:, np.newaxis], 
            loc=self.obs_mean[np.newaxis, :], 
            scale=self.obs_std[np.newaxis, :])

        self.log_evidence = logpsi

    @classmethod
    def _make_duration_dist(self):
        dvec = np.arange(self.D)
        logpdf = stats.norm.logpdf(dvec[np.newaxis, :], 
            loc=self.obs_mean[:, np.newaxis],
            scale=self.obs_std[:, np.newaxis])

        # normalize
        logpdf -= np.log(np.sum(np.exp(logpdf), axis=1, keepdims=True))

        self.dvec = dvec
        self.logpd = logpdf

    def test_invalid_probs_raise_error(self):
        bad_log_A = np.log(self.A)
        bad_log_A[0] = np.abs(bad_log_A[0])  # means entry in A > 1
        bad_log_pi = np.log(self.pi)
        bad_log_pi[0] = np.abs(bad_log_pi[0])  # means entry in A > 1
        assert_raises(ValueError, fb.fb_infer, 
            bad_log_A, np.log(self.pi), self.log_evidence,
            self.dvec, self.logpd)
        assert_raises(ValueError, fb.fb_infer, 
            np.log(self.A), bad_log_pi, self.log_evidence,
            self.dvec, self.logpd)

    def test_calc_B(self):
        B = np.empty((self.T, self.K, self.D))
        cum_log_psi = np.empty((self.T, self.K))
        fb._calc_B(self.dvec, self.log_evidence, B, cum_log_psi)

        # first, check cum_log_psi
        npt.assert_allclose(cum_log_psi[3], 
            np.cumsum(self.log_evidence, axis=0)[3])

        # check a few entries in B
        didx = 2
        this_d = self.dvec[didx]
        this_t = 5
        start = max(0, this_t - this_d + 1)
        npt.assert_allclose(B[this_t, :, didx], 
            cum_log_psi[this_t] - cum_log_psi[start - 1])

        this_t = this_d
        start = max(0, this_t - this_d + 1)
        npt.assert_allclose(B[this_t, :, didx], 
            cum_log_psi[this_t] - cum_log_psi[start - 1])

        this_t = this_d - 1
        start = max(0, this_t - this_d + 1)
        npt.assert_allclose(B[this_t, :, didx], 
            cum_log_psi[this_t])

if __name__ == '__main__':
    np.random.rand(12345)

    TFB = Test_Forwards_Backwards()
    TFB.setup_hsmm()