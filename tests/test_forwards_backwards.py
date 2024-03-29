"""
Tests for Forwards-Backwards inference model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_true, set_trace
import numpy as np
import scipy.stats as stats
import numpy.testing as npt
import gamma_poisson as gp
import forward_backward as fb

class Test_Forwards_Backwards:
    @classmethod
    def setup_class(self):
        """
        Make some sample Markov chain data to run inference on.
        """

        np.random.seed(12345)

        self._setup_constants()

        self._make_transition_probs()

        self._make_initial_states()

        self._make_chain()

        # make test case for forwards-backwards
        self._setup_fb_data()

        # do a little preprocessing (calculate emission probabilities)
        self._calc_emission_probs()

    @classmethod
    def _setup_constants(self):
        """
        Set variables that determine the problem size.
        """
        self.U = 10  # units
        self.T = 500  # time points
        self.K = 5  # categories
        self.dt = 1 / 30  # time length of observation (in s)

    @classmethod
    def _make_transition_probs(self):
        """
        Make a vector of transition probabilities for each category:
        2 rows x K columns
        Row 0 contains 0 -> 1 transition probability
        Row 1 contains 1 -> transition probability
        """ 
        # want 0 -> to be rare, 1 -> 1 to be a little less so
        v0 = stats.beta.rvs(1, 200, size=self.K)
        v1 = stats.beta.rvs(100, 1, size=self.K)
        self.trans_probs = np.vstack([v0, v1]).T 

    @classmethod
    def _make_initial_states(self):
        """
        pick initial states for each category
        """
        z0_probs = stats.beta.rvs(50, 50, size=self.K)
        self.z0 = stats.bernoulli.rvs(z0_probs)

    @classmethod
    def _make_chain(self):
        """
        Make a Markov chain by using the transition probabilities.
        """

        # make the chain by evolving each category through time
        chain = np.empty((self.K, self.T), dtype='int')

        # initialize
        chain[:, 0] = self.z0

        for t in xrange(1, self.T):
            p_trans_to_1 = self.trans_probs[range(self.K), chain[:, t - 1]]
            chain[:, t] = stats.bernoulli.rvs(p_trans_to_1)
            
        # add a baseline that's always turned on
        chain[0, :] = 1

        self.chain = chain

    @classmethod
    def _setup_fb_data(self):
        idx = 1  # which chain to test
        self.U = 10  # number of units for test case
        self.z_test = self.chain[idx]

        # rates for each unit
        mu0 = np.random.rand(self.U)
        mu = 5 * np.random.rand(self.U)
        mu_stack = np.dstack([mu0, mu0 + mu])
        self.mu_test = mu_stack[0]

        # series of observations: ground truth
        rates = mu0 + np.outer(self.z_test, mu)
        self.N_test = stats.poisson.rvs(rates)

        # transition matrix
        to_1 = np.expand_dims(self.trans_probs[idx, :], 0)
        self.A_test = np.r_[1 - to_1, to_1]

        # initial belief about states
        self.pi0_test = np.array([0.5, 0.5])

    @classmethod
    def _calc_emission_probs(self):
        logpsi = np.sum(self.N_test[:, :, np.newaxis] * 
            np.log(self.mu_test[np.newaxis, ...]), axis=1)
        self.psi_test = np.exp(logpsi)

    def test_class_setup(self):
        assert_equals(self.chain.shape, (self.K, self.T))
        assert_equals(np.max(self.chain), 1)
        assert_equals(np.min(self.chain), 0)
        npt.assert_allclose(self.chain, np.around(self.chain))
        assert_equals(self.z_test.shape, (self.T,))
        assert_equals(self.mu_test.shape, (self.U, 2))
        assert_equals(self.N_test.shape, (self.T, self.U))
        assert_equals(self.A_test.shape, (2, 2))
        assert_equals(self.pi0_test.shape, (2,))

    def test_fb_returns_consistent(self):
        gamma, logZ, Xi = fb.fb_infer(self.A_test, self.pi0_test, self.psi_test)
        assert_equals(gamma.shape, (self.T, 2))
        npt.assert_allclose(np.sum(gamma, 1), np.ones((self.T,)))
        assert_equals(logZ.shape, ())
        assert_equals(Xi.shape, (self.T - 1, 2, 2))
        npt.assert_allclose(np.sum(Xi, axis=(1, 2)), np.ones((self.T - 1, )))

    def test_rescaling_psi_compensated_by_Z(self):
        offset = np.random.normal(size=self.T)
        psi_r = np.exp(np.log(self.psi_test) + offset[:, np.newaxis])
        gamma, logZ, Xi = gp.fb_infer(self.A_test, self.pi0_test, self.psi_test)

        gammar, logZr, Xir = gp.fb_infer(self.A_test, self.pi0_test, psi_r)

        npt.assert_allclose(gammar, gamma)
        npt.assert_allclose(logZr, logZ + np.sum(offset))
        npt.assert_allclose(Xi, Xir)

    def test_rescaling_pi_compensated_by_Z(self):
        offset = np.random.normal()
        pi_r = np.exp(np.log(self.pi0_test) + offset)
        gamma, logZ, Xi = gp.fb_infer(self.A_test, self.pi0_test, self.psi_test)

        gammar, logZr, Xir = gp.fb_infer(self.A_test, pi_r, self.psi_test)

        npt.assert_allclose(gammar, gamma)
        npt.assert_allclose(logZr, logZ + offset)
        npt.assert_allclose(Xi, Xir)

    def test_rescaling_A_compensated_by_Z(self):
        offset = np.random.normal()
        A_r = np.exp(np.log(self.A_test) + offset)
        gamma, logZ, Xi = gp.fb_infer(self.A_test, self.pi0_test, self.psi_test)

        gammar, logZr, Xir = gp.fb_infer(A_r, self.pi0_test, self.psi_test)

        npt.assert_allclose(gammar, gamma)
        npt.assert_allclose(logZr, logZ + (self.T - 1) * offset)
        npt.assert_allclose(Xi, Xir)

    def test_entropy_positive(self):
        # we want H = -E_q[log q] > 0
        # this corresponds to log Z - E_q[log likelihood] > 0
        gamma, logZ, Xi = fb.fb_infer(self.A_test, self.pi0_test, self.psi_test)
        emission_piece = np.sum(gamma * np.log(self.psi_test))
        initial_piece = np.sum(gamma[0] * np.log(self.pi0_test))
        transition_piece = np.sum(Xi * np.log(self.A_test))
        LL = emission_piece + initial_piece + transition_piece
        assert_true(logZ - LL >= 0)

    def test_entropy_positive_subadditive_pars(self):
        # same test for positive entropy, but with subadditive A and pi
        # offset = -5
        # log_pi_r = np.log(self.pi0_test) + offset
        # log_A_r = np.log(self.A_test) + offset
        log_pi_r = np.log(self.pi0_test) - np.random.random(size=self.pi0_test.shape)
        log_A_r = np.log(self.A_test) - np.random.random(size=self.A_test.shape)

        gamma, logZ, Xi = fb.fb_infer(np.exp(log_A_r), np.exp(log_pi_r),
         self.psi_test)
        emission_piece = np.sum(gamma * np.log(self.psi_test))
        initial_piece = np.sum(gamma[0] * log_pi_r)
        transition_piece = np.sum(Xi * log_A_r)
        LL = emission_piece + initial_piece + transition_piece
        assert_true(logZ - LL >= 0)

    def fb_infer_integration_test(self):
        gamma, logZ, Xi = fb.fb_infer(self.A_test, self.pi0_test, self.psi_test)
        npt.assert_allclose(self.z_test.astype('float')[5:], 
            gamma[5:, 1], atol=1e-3)
        
