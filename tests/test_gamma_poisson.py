"""
Tests for Gamma Poisson model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, set_trace
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
        assert_is_instance(gpm.Lvalues, list)

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

    def test_can_set_inits(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        mutest = np.random.rand(self.K, self.U)
        alphatest = np.random.rand(self.K, self.U)
        betatest = np.random.rand(self.K, self.U)
        gamma1test = np.random.rand(2, self.K)
        gamma2test = np.random.rand(2, self.K)
        delta1test = np.random.rand(self.K)
        delta2test = np.random.rand(self.K)
        xitest = np.random.rand(self.T, self.K)
        xitest_normed = xitest.copy()
        xitest_normed[:, 0] = 1
        Xitest = np.random.rand(self.T  - 1, self.K, 2, 2)

        gpm.set_inits(mu=mutest, alpha=alphatest, beta=betatest, 
            gamma1=gamma1test, gamma2=gamma2test, delta1=delta1test,
            delta2=delta2test, xi=xitest, Xi=Xitest)
        npt.assert_array_equal(gpm.mu, mutest)
        npt.assert_array_equal(gpm.alpha, alphatest)
        npt.assert_array_equal(gpm.beta, betatest)
        npt.assert_array_equal(gpm.gamma1, gamma1test)
        npt.assert_array_equal(gpm.gamma2, gamma2test)
        npt.assert_array_equal(gpm.delta1, delta1test)
        npt.assert_array_equal(gpm.delta2, delta2test)
        npt.assert_array_equal(gpm.xi, xitest_normed)

        # Xi will have been normalized, and since baseline category is 
        # included by default, Xi[:, k] = [[0, 0] , [0, 1]]
        Xinorm = Xitest / np.sum(Xitest, axis=(-1, -2), keepdims=True)
        Xinorm[:, 0] = 0
        Xinorm[:, 0, 1, 1] = 1
        npt.assert_allclose(gpm.Xi, Xinorm)
        npt.assert_allclose(np.sum(gpm.Xi, axis=(-1, -2)), 
            np.ones((self.T - 1, self.K)))

    def test_can_set_inits_no_baseline(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt, include_baseline=False)
        mutest = np.random.rand(self.K, self.U)
        alphatest = np.random.rand(self.K, self.U)
        betatest = np.random.rand(self.K, self.U)
        gamma1test = np.random.rand(2, self.K)
        gamma2test = np.random.rand(2, self.K)
        delta1test = np.random.rand(self.K)
        delta2test = np.random.rand(self.K)
        xitest = np.random.rand(self.T, self.K)
        xitest_normed = xitest.copy()
        xitest_normed[:, 0] = 1
        Xitest = np.random.rand(self.T  - 1, self.K, 2, 2)

        gpm.set_inits(mu=mutest, alpha=alphatest, beta=betatest, 
            gamma1=gamma1test, gamma2=gamma2test, delta1=delta1test,
            delta2=delta2test, xi=xitest, Xi=Xitest)
        npt.assert_array_equal(gpm.mu, mutest)
        npt.assert_array_equal(gpm.alpha, alphatest)
        npt.assert_array_equal(gpm.beta, betatest)
        npt.assert_array_equal(gpm.gamma1, gamma1test)
        npt.assert_array_equal(gpm.gamma2, gamma2test)
        npt.assert_array_equal(gpm.delta1, delta1test)
        npt.assert_array_equal(gpm.delta2, delta2test)
        npt.assert_array_equal(gpm.xi, xitest)

        # Xi will have been normalized
        Xinorm = Xitest / np.sum(Xitest, axis=(-1, -2), keepdims=True)
        npt.assert_allclose(gpm.Xi, Xinorm)
        npt.assert_allclose(np.sum(gpm.Xi, axis=(-1, -2)), 
            np.ones((self.T - 1, self.K)))

    def test_no_arguments_sets_default_inits(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        gpm.set_inits()
        assert_equals(gpm.mu.shape, (self.K, self.U))
        assert_equals(gpm.alpha.shape, (self.K, self.U))
        assert_equals(gpm.beta.shape, (self.K, self.U))
        assert_equals(gpm.gamma1.shape, (2, self.K))
        assert_equals(gpm.gamma2.shape, (2, self.K))
        assert_equals(gpm.delta1.shape, (self.K,))
        assert_equals(gpm.delta2.shape, (self.K,))

    def test_invalid_init_shapes_raises_error(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        assert_raises(ValueError, gpm.set_inits, 
            mu=np.ones((self.K + 1, self.U)))
        assert_raises(ValueError, gpm.set_inits, 
            alpha=np.ones((self.K, self.U, 1)))
        assert_raises(ValueError, gpm.set_inits, 
            beta=np.ones((self.K,)))
        assert_raises(ValueError, gpm.set_inits, 
            gamma1=np.ones((self.K + 1, self.U)))
        assert_raises(ValueError, gpm.set_inits, 
            gamma2=np.ones((1, self.U)))
        assert_raises(ValueError, gpm.set_inits, 
            delta1=np.ones((1, self.K)))
        assert_raises(ValueError, gpm.set_inits, 
            delta2=np.ones((2, self.K, 1)))
        assert_raises(ValueError, gpm.set_inits, 
            xi=np.ones((self.T, self.K, 1)))

    def test_F_prod(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        z = np.random.rand(5, 8)
        w = np.random.rand(8, 3)
        assert_equals(gpm.F_prod(z, w).shape, (5, 8, 3))
        npt.assert_allclose(gpm.F_prod(z, w), 
            np.exp(gpm.F_prod(z, w, log=True)))

        # test whether we can get the same entry in every element by
        # multiplying through by the one part each lacks
        unnorm = 1 - z[..., np.newaxis] + z[..., np.newaxis] * w
        allprod = unnorm * gpm.F_prod(z, w)
        npt.assert_allclose(allprod[0, 0, 0], allprod[0, 1, 0])
        npt.assert_allclose(allprod[-1, 0, 2], allprod[-1, -1, 2])
        npt.assert_allclose(allprod[:, 0, :], gpm.F_prod(z, w, exclude=False))

    def test_calc_A(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        gpm.set_inits()
        A = gpm.calc_A()
        assert_equals(A.shape, (2, 2, self.K))
        npt.assert_allclose(np.sum(A, 0), np.ones((2, self.K)))

    def test_L(self):
        # just test that we can run the function without errors
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        gpm.set_priors().set_inits().set_data(self.N)
        L0 = gpm.L()
        assert_equals(L0.shape, ())

    def test_update_chain_rates(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        gpm.set_priors().set_inits().set_data(self.N).iterate()
        L0 = gpm.L()
        L1 = gpm.update_chain_rates(0).L()
        L2 = gpm.update_chain_rates(0).L()
        assert_true(L1 < np.inf)
        assert_equals(L1, L2)
        assert_true(L0 <= gpm.L())

    def test_update_chain_states(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        gpm.set_priors().set_inits().set_data(self.N).iterate()
        L0 = gpm.L()
        L1 = gpm.update_chain_states(0).L()
        L2 = gpm.update_chain_states(0).L()
        assert_true(np.max(gpm.xi[:, 0] <= 1))
        assert_true(np.min(gpm.xi[:, 0] >= 0))
        assert_true(L1 < np.inf)
        assert_equals(L1, L2)
        assert_true(L0 <= gpm.L())

    def test_update_chain_pars(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        gpm.set_priors().set_inits().set_data(self.N).iterate()
        L0 = gpm.L()
        gpm.update_chain_pars(1).L()
        L2 = gpm.update_chain_states(1).L()
        L3 = gpm.update_chain_pars(1).L()
        L4 = gpm.update_chain_pars(1).L()
        assert_true(L2 < np.inf)
        assert_equals(L3, L4)
        assert_true(L0 <= gpm.L())

    def test_iteration_increases_L(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        gpm.set_priors().set_inits().set_data(self.N).iterate()
        Lvals = [gpm.L()]
        for _ in xrange(5):
            gpm.iterate()
            Lvals.append(gpm.L())
        assert_true(np.all(np.diff(Lvals) >= 0))

    def test_inference_appends_to_Lvalues(self):
        gpm = gp.GPModel(self.T, self.K, self.U, self.dt)
        assert_equals(len(gpm.Lvalues), 0)
        gpm.set_priors().set_inits().set_data(self.N)
        gpm.do_inference(tol=1)
        assert_equals(len(gpm.Lvalues), 1)
