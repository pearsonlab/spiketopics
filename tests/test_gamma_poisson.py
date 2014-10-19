"""
Tests for Gamma Poisson model.
"""
from nose.tools import assert_equals, assert_is_instance, assert_raises
import numpy as np
import scipy.stats as stats
import numpy.testing as npt
import gamma_poisson as gp

class Test_Gamma_Poisson:
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

    def test_can_instantiate_model_object(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        assert_is_instance(gpm, gp.GPModel)
        assert_equals(gpm.U, self.U)
        assert_equals(gpm.T, self.T)
        assert_equals(gpm.K, self.K)
        assert_equals(gpm.dt, self.dt)

    def test_can_set_priors(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        cctest = np.random.rand(self.K, self.U)
        ddtest = np.random.rand(self.K, self.U)
        nu1test = np.random.rand(2, self.K)
        nu2test = np.random.rand(2, self.K)
        rho1test = np.random.rand(self.K)
        rho2test = np.random.rand(self.K)
        gpm.set_priors(cc=cctest, dd=ddtest, nu1=nu1test, nu2=nu2test,
            rho1=rho1test, rho2=rho2test)
        npt.assert_array_equal(gpm.cc, cctest)
        npt.assert_array_equal(gpm.dd, ddtest)
        npt.assert_array_equal(gpm.nu1, nu1test)
        npt.assert_array_equal(gpm.nu2, nu2test)
        npt.assert_array_equal(gpm.rho1, rho1test)
        npt.assert_array_equal(gpm.rho2, rho2test)

    def test_no_arguments_sets_default_priors(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        gpm.set_priors()
        assert_equals(gpm.cc.shape, (self.K, self.U))
        assert_equals(gpm.dd.shape, (self.K, self.U))
        assert_equals(gpm.nu1.shape, (2, self.K))
        assert_equals(gpm.nu2.shape, (2, self.K))
        assert_equals(gpm.rho1.shape, (self.K,))
        assert_equals(gpm.rho2.shape, (self.K,))

    def test_invalid_prior_shapes_raises_error(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        assert_raises(ValueError, gpm.set_priors, 
            cc=np.ones((self.K + 1, self.U)))
        assert_raises(ValueError, gpm.set_priors, 
            dd=np.ones((self.K + 1, self.U)))
        assert_raises(ValueError, gpm.set_priors, 
            nu1=np.ones((self.K + 1, self.U)))
        assert_raises(ValueError, gpm.set_priors, 
            nu2=np.ones((1, self.U)))
        assert_raises(ValueError, gpm.set_priors, 
            rho1=np.ones((1, self.K)))
        assert_raises(ValueError, gpm.set_priors, 
            rho2=np.ones((2, self.K, 1)))



