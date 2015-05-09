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

    def test_calc_B(self):
        B = np.empty((self.T + 1, self.K, self.Ddim))
        cum_log_psi = np.empty((self.T + 1, self.K))
        fb._calc_B(self.dvec, self.log_evidence, B, cum_log_psi)

        # first, check cum_log_psi
        npt.assert_allclose(cum_log_psi[1], self.log_evidence[0])
        npt.assert_allclose(cum_log_psi[3], 
            np.cumsum(self.log_evidence, axis=0)[2])
        assert_true(np.all(np.isfinite(cum_log_psi)))

        # check B in general
        assert_true(np.all(np.isfinite(B)))

        # check a few entries in B
        npt.assert_allclose(B[1, 2], cum_log_psi[1, 2])

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

    def test_forward(self):
        B = np.empty((self.T, self.K, self.Ddim))
        cum_log_psi = np.empty((self.T, self.K))
        fb._calc_B(self.dvec, self.log_evidence, B, cum_log_psi)

        alpha = np.empty((self.T + 1, self.K))
        alpha_star = np.empty((self.T + 1, self.K))

        fb._forward(alpha, alpha_star, np.log(self.A), np.log(self.pi),
            B, self.dvec, self.logpd)

        assert(np.all(np.isfinite(alpha[-1])))
        assert(np.all(np.isfinite(alpha_star[-1])))

        # sum over all final states should give the same p(evidence), whether
        # or not we use alpha or alpha_star
        fin = -np.inf
        fin_star = -np.inf
        for j in xrange(self.K):
            fin = np.logaddexp(fin, alpha[-1, j])
            fin_star = np.logaddexp(fin_star, alpha_star[-1, j])
        npt.assert_allclose(fin, fin_star)

    def test_backward(self):
        B = np.empty((self.T, self.K, self.Ddim))
        cum_log_psi = np.empty((self.T, self.K))
        fb._calc_B(self.dvec, self.log_evidence, B, cum_log_psi)

        beta = np.empty((self.T + 1, self.K))
        beta_star = np.empty((self.T + 1, self.K))
        fb._backward(beta, beta_star, np.log(self.A), B, self.dvec, self.logpd)

        npt.assert_allclose(beta[-1], 1)
        npt.assert_allclose(beta_star[-1], 1)
        assert(np.all(np.isfinite(beta[0])))
        assert(np.all(np.isfinite(beta_star[0])))

    def test_logZ(self):
        B = np.empty((self.T, self.K, self.Ddim))
        cum_log_psi = np.empty((self.T, self.K))
        fb._calc_B(self.dvec, self.log_evidence, B, cum_log_psi)

        alpha = np.empty((self.T + 1, self.K))
        alpha_star = np.empty((self.T + 1, self.K))
        fb._forward(alpha, alpha_star, np.log(self.A), np.log(self.pi),
            B, self.dvec, self.logpd)
        logZ = fb._calc_logZ(alpha)
        assert(np.isfinite(logZ))

    def test_posterior(self):
        B = np.empty((self.T, self.K, self.Ddim))
        cum_log_psi = np.empty((self.T, self.K))
        fb._calc_B(self.dvec, self.log_evidence, B, cum_log_psi)

        # forward
        alpha = np.empty((self.T + 1, self.K))
        alpha_star = np.empty((self.T + 1, self.K))
        fb._forward(alpha, alpha_star, np.log(self.A), np.log(self.pi),
            B, self.dvec, self.logpd)

        # backward
        beta = np.empty((self.T + 1, self.K))
        beta_star = np.empty((self.T + 1, self.K))
        fb._backward(beta, beta_star, np.log(self.A), B, self.dvec, self.logpd)

        # posterior
        gamma = np.empty((self.T + 1, self.K))
        gamma_star = np.empty((self.T + 1, self.K))
        post = np.empty((self.T + 1, self.K))
        fb._calc_posterior(alpha, alpha_star, beta, beta_star, 
            gamma, gamma_star, post) 
        npt.assert_allclose(np.sum(post, 1)[1:], 1.0)

    def test_two_slice(self):
        B = np.empty((self.T, self.K, self.Ddim))
        cum_log_psi = np.empty((self.T, self.K))
        fb._calc_B(self.dvec, self.log_evidence, B, cum_log_psi)

        # forward
        alpha = np.empty((self.T + 1, self.K))
        alpha_star = np.empty((self.T + 1, self.K))
        fb._forward(alpha, alpha_star, np.log(self.A), np.log(self.pi),
            B, self.dvec, self.logpd)

        # backward
        beta = np.empty((self.T + 1, self.K))
        beta_star = np.empty((self.T + 1, self.K))
        fb._backward(beta, beta_star, np.log(self.A), B, self.dvec, self.logpd)

        # posterior
        gamma = np.empty((self.T + 1, self.K))
        gamma_star = np.empty((self.T + 1, self.K))
        post = np.empty((self.T + 1, self.K))
        fb._calc_posterior(alpha, alpha_star, beta, beta_star, 
            gamma, gamma_star, post) 

        # two-slice marginals
        Xi = np.empty((self.T - 1, self.K, self.K))
        fb._calc_two_slice(alpha, beta_star, np.log(self.A), Xi)
        npt.assert_allclose(np.sum(np.exp(Xi), axis=(1, 2)), 1.)
        # we should have gamma_star = sum_z p(z', z|transition) 
        # = sum_z Xi_{z'z}
        npt.assert_allclose(gamma_star[1:-1], np.logaddexp.reduce(Xi, 2), 
            atol=1e-10)
        npt.assert_allclose(gamma[1:-1], np.logaddexp.reduce(Xi, 1), 
            atol=1e-10)

    def test_estimate_duration_dist(self):
        B = np.empty((self.T, self.K, self.Ddim))
        cum_log_psi = np.empty((self.T, self.K))
        fb._calc_B(self.dvec, self.log_evidence, B, cum_log_psi)

        # forward
        alpha = np.empty((self.T + 1, self.K))
        alpha_star = np.empty((self.T + 1, self.K))
        fb._forward(alpha, alpha_star, np.log(self.A), np.log(self.pi),
            B, self.dvec, self.logpd)

        # backward
        beta = np.empty((self.T + 1, self.K))
        beta_star = np.empty((self.T + 1, self.K))
        fb._backward(beta, beta_star, np.log(self.A), B, self.dvec, self.logpd)

        # posterior
        gamma = np.empty((self.T + 1, self.K))
        gamma_star = np.empty((self.T + 1, self.K))
        post = np.empty((self.T + 1, self.K))
        fb._calc_posterior(alpha, alpha_star, beta, beta_star, 
            gamma, gamma_star, post) 

        # sufficient stats for p(d|z)
        logpd_hat = np.empty((self.K, self.Ddim))
        fb._estimate_duration_dist(alpha_star, beta, B, self.dvec, 
            self.logpd, logpd_hat,)
        npt.assert_allclose(np.logaddexp.reduce(logpd_hat, 1), 
            np.logaddexp.reduce((alpha + beta)[1:], 0), atol=0.1)

    def test_rescaling_psi_compensated_by_Z(self):
        offset = np.random.normal(size=self.T)
        psi_r = self.log_evidence + offset[:, np.newaxis]
        xi, logZ, Xi, C = fb.fb_infer(np.log(self.A), np.log(self.pi), 
            self.log_evidence, self.dvec, self.logpd)

        xi_r, logZ_r, Xi_r, C_r = fb.fb_infer(np.log(self.A), np.log(self.pi), 
            psi_r, self.dvec, self.logpd)

        npt.assert_allclose(xi, xi_r)
        npt.assert_allclose(logZ_r, logZ + np.sum(offset))
        npt.assert_allclose(Xi, Xi_r)
        
if __name__ == '__main__':
    np.random.rand(12345)

    TFB = Test_Forwards_Backwards()
    TFB.setup_hsmm()