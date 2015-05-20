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
        np.random.seed(12345)

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
        self.M = 4  # levels of hidden state
        self.D = 50

    @classmethod
    def _make_transition_probs(self):
            """
            Make a Markov transition matrix and initial state vector.
            Columns of A sum to 1, so A acts to the right.
            """ 
            lo, hi = 1, 20
            rows = []
            for _ in xrange(self.M):
                alpha = stats.randint.rvs(lo, hi, size=self.M)
                row = stats.dirichlet.rvs(alpha)
                rows.append(row)
            self.A = np.vstack(rows).T

            alpha = stats.randint.rvs(lo, hi, size=self.M)
            self.pi = stats.dirichlet.rvs(alpha).squeeze()

    @classmethod
    def _make_chain(self):
        """
        Make a Markov chain by using the transition probabilities.
        """

        # make the chain by evolving each category through time
        chain = np.empty((self.M, self.T), dtype='int')

        # pick pars for duration distribution
        mm = 10 * np.random.rand(self.M) # mean
        ss = 3 * np.random.rand(self.M) # standard deviation

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
        mu = 10 * np.random.rand(self.M)
        sig = 2 * np.random.rand(self.M)

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
        dvec = np.arange(1, self.D)
        logpdf = stats.norm.logpdf(dvec[np.newaxis, :], 
            loc=self.dur_mean[:, np.newaxis],
            scale=self.dur_std[:, np.newaxis])

        # normalize
        logpdf -= np.logaddexp.reduce(logpdf, axis=1, keepdims=True)

        self.Ddim = len(dvec)
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

    def test_forward(self):
        logalpha = np.empty((self.T, self.M, self.Ddim)) 
        logZ = np.empty(self.T,)
        logE = np.empty((self.T, self.M))
        logS = np.empty((self.T, self.M))

        fb._forward(logalpha, logZ, logE, logS, np.log(self.pi), 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        # alpha finite
        assert(np.all(np.isfinite(logalpha)))

        # alpha normalized
        loggamma = np.logaddexp.reduce(logalpha, 2)
        npt.assert_allclose(np.exp(np.logaddexp.reduce(loggamma, 1)), 1.)

        # Z correct
        lZ = np.logaddexp.reduce(loggamma + self.log_evidence, 1)
        npt.assert_allclose(logZ, lZ)

        # S and E compatible
        lS = np.logaddexp.reduce(logS, 1)
        lE = np.logaddexp.reduce(logE, 1)
        npt.assert_allclose(lS, lE)

    def test_backward(self):
        logalpha = np.empty((self.T, self.M, self.Ddim)) 
        logbeta = np.empty((self.T, self.M, self.Ddim)) 
        logZ = np.empty(self.T,)
        logE = np.empty((self.T, self.M))
        logEstar = np.empty((self.T, self.M))
        logS = np.empty((self.T, self.M))
        logSstar = np.empty((self.T, self.M))

        fb._forward(logalpha, logZ, logE, logS, np.log(self.pi), 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        fb._backward(logbeta, logZ, logEstar, logSstar, 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        # beta finite
        assert(np.all(np.isfinite(logbeta)))

    def test_posterior(self):
        logalpha = np.empty((self.T, self.M, self.Ddim)) 
        logbeta = np.empty((self.T, self.M, self.Ddim)) 
        logZ = np.empty(self.T,)
        logE = np.empty((self.T, self.M))
        logEstar = np.empty((self.T, self.M))
        logS = np.empty((self.T, self.M))
        logSstar = np.empty((self.T, self.M))
        logxi = np.empty((self.T, self.M))

        fb._forward(logalpha, logZ, logE, logS, np.log(self.pi), 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        fb._backward(logbeta, logZ, logEstar, logSstar, 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        fb._posterior(logalpha, logbeta, logxi)

        # posterior sums to 1
        norm = np.logaddexp.reduce(logxi, 1)
        npt.assert_allclose(np.exp(norm), 1.)

    def test_two_slice(self):
        logalpha = np.empty((self.T, self.M, self.Ddim)) 
        logbeta = np.empty((self.T, self.M, self.Ddim)) 
        logZ = np.empty(self.T,)
        logE = np.empty((self.T, self.M))
        logEstar = np.empty((self.T, self.M))
        logS = np.empty((self.T, self.M))
        logSstar = np.empty((self.T, self.M))
        logXi = np.empty((self.T - 1, self.M, self.M))

        fb._forward(logalpha, logZ, logE, logS, np.log(self.pi), 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        fb._backward(logbeta, logZ, logEstar, logSstar, 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        fb._two_slice(logE, logEstar, np.log(self.A), logXi)

        assert(np.all(np.isfinite(logXi)))


    def test_sequence_entry(self):
        logalpha = np.empty((self.T, self.M, self.Ddim)) 
        logbeta = np.empty((self.T, self.M, self.Ddim)) 
        logZ = np.empty(self.T,)
        logE = np.empty((self.T, self.M))
        logEstar = np.empty((self.T, self.M))
        logS = np.empty((self.T, self.M))
        logSstar = np.empty((self.T, self.M))
        logC = np.empty((self.T, self.M, self.Ddim))

        fb._forward(logalpha, logZ, logE, logS, np.log(self.pi), 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        fb._backward(logbeta, logZ, logEstar, logSstar, 
            np.log(self.A), self.log_evidence, self.dvec, self.logpd)

        fb._sequence_entry(logS, self.logpd, np.log(self.pi), 
            logbeta, logC)

        assert(np.all(np.isfinite(logC)))

    def test_rescaling_psi_compensated_by_Z(self):
        offset = np.random.normal(size=self.T)
        psi_r = self.log_evidence + offset[:, np.newaxis]
        xi, logZ, Xi, C = fb.fb_infer(np.log(self.A), np.log(self.pi), 
            self.log_evidence, self.dvec, self.logpd)

        xi_r, logZ_r, Xi_r, C_r = fb.fb_infer(np.log(self.A), np.log(self.pi), 
            psi_r, self.dvec, self.logpd)

        npt.assert_allclose(xi, xi_r, atol=1e-10)
        npt.assert_allclose(logZ_r, logZ + np.sum(offset))
        npt.assert_allclose(Xi, Xi_r)
        npt.assert_allclose(C, C_r)
        
    def test_rescaling_pd_compensated_by_Z(self):
        rescale = 0.77

        xi, logZ, Xi, C = fb.fb_infer(np.log(self.A), np.log(self.pi), 
            self.log_evidence, self.dvec, self.logpd)

        xi_r, logZ_r, Xi_r, C_r = fb.fb_infer(np.log(self.A), np.log(self.pi), 
            self.log_evidence, self.dvec, self.logpd + np.log(rescale))

        npt.assert_allclose(xi, xi_r, atol=1e-10)
        npt.assert_allclose(logZ_r, logZ + np.sum(C * np.log(rescale)))
        npt.assert_allclose(Xi, Xi_r)
        npt.assert_allclose(C, C_r)

    def test_rescaling_A_compensated_by_Z(self):
        rescale = 0.77

        xi, logZ, Xi, C = fb.fb_infer(np.log(self.A), np.log(self.pi), 
            self.log_evidence, self.dvec, self.logpd)

        xi_r, logZ_r, Xi_r, C_r = fb.fb_infer(np.log(self.A) + 
            np.log(rescale), np.log(self.pi), 
            self.log_evidence, self.dvec, self.logpd)

        npt.assert_allclose(xi, xi_r, atol=1e-10)
        npt.assert_allclose(logZ_r, logZ + np.log(rescale) * np.sum(Xi))
        npt.assert_allclose(Xi, Xi_r)
        npt.assert_allclose(C, C_r)

if __name__ == '__main__':
    np.random.rand(12345)

    TFB = Test_Forwards_Backwards()
    TFB.setup_hsmm()