"""
Tests for Gamma-Poisson model.
"""
from nose.tools import assert_equals, assert_true
import numpy as np
import scipy.stats as stats
import numpy.testing as npt
import gamma_poisson as gp

class Test_Forwards_Backwards_Integration():
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

        self._make_firing_rates()

        # draw from Poisson
        self.N = stats.poisson.rvs(self.fr)

        # make smaller test case for forwards-backwards
        self._setup_fb_data()


    @classmethod
    def _setup_constants(self):
        """
        Set variables that determine the problem size.
        """
        self.U = 100  # units
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
    def _make_firing_rates(self):
        """
        Make matrix of firing rates (categories x units)
        """        

        aa = 3, 1
        bb = 2, 1
        lam = stats.gamma.rvs(a=aa[1], scale=bb[1], size=(self.K, self.U))

        # baselines should follow a different distribution
        lam[0, :] = stats.gamma.rvs(a=aa[0], scale=bb[0], size=self.U)   

        # calculate rate for each time
        fr = np.exp(self.chain.T.dot(np.log(lam))) * self.dt
        self.fr = fr + 1e-5  # in case we get exactly 0

    @classmethod
    def _setup_fb_data(self):
        idx = 1  # which chain to test
        self.num_units = 10  # number of units for test case
        self.ztest = self.chain[idx]

        # rates for each unit
        mu0 = np.random.rand(self.num_units)
        mu = 5 * np.random.rand(self.num_units)
        mu_stack = np.dstack([mu0, mu0 + mu])
        self.mu_test = mu_stack[0]

        # series of observations: ground truth
        rates = mu0 + np.outer(self.ztest, mu)
        self.N_test = stats.poisson.rvs(rates)

        # transition matrix
        to_1 = np.expand_dims(self.trans_probs[idx, :], 0)
        self.A_test = np.r_[1 - to_1, to_1]

        # initial belief about states
        self.pi0_test = np.array([0.5, 0.5])

    def test_class_setup(self):
        assert_equals(self.chain.shape, (self.K, self.T))
        assert_equals(np.max(self.chain), 1)
        assert_equals(np.min(self.chain), 0)
        npt.assert_allclose(self.chain, np.around(self.chain))
        npt.assert_array_less(0, self.fr)
        npt.assert_string_equal(self.N.dtype.kind, 'i')
        assert_equals(self.ztest.shape, (self.T,))
        assert_equals(self.mu_test.shape, (self.num_units, 2))
        assert_equals(self.N_test.shape, (self.T, self.num_units))
        assert_equals(self.A_test.shape, (2, 2))
        assert_equals(self.pi0_test.shape, (2,))

    def test_observation_probs_consistent(self):
        psi = gp.calculate_observation_probs(self.N_test, self.mu_test)
        assert_equals(psi.shape, (self.T, self.mu_test.shape[-1]))
        npt.assert_array_equal(psi >= 0, np.ones_like(psi, dtype='bool'))

    def test_observation_probs_corrects_impossible_baseline(self):
        # set up a case were N is nonzero probability, baseline is 
        # 0, and function corrects this 
        Ntest = self.N_test.copy()
        Ntest[0] = 1 
        mutest = self.mu_test.copy()
        mutest[0, 0] = 0
        psi = gp.calculate_observation_probs(Ntest, mutest)
        npt.assert_allclose(psi[0, :], np.array([0.0, 1.0]), atol=1e-10)        
    def test_fb_returns_consistent(self):
        gamma, logZ, Xi = gp.fb_infer(self.N_test, self.mu_test, self.A_test, 
            self.pi0_test) 
        assert_equals(gamma.shape, (self.T, 2))
        npt.assert_allclose(np.sum(gamma, 1), np.ones((self.T,)))
        assert_equals(logZ.shape, ())
        assert_true(logZ >= 0)
        assert_equals(Xi.shape, (self.T - 1, 2, 2))
        npt.assert_allclose(np.sum(Xi, axis=(1, 2)), np.ones((self.T - 1, )))