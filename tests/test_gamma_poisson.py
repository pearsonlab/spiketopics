"""
Tests for Gamma Poisson model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace
import numpy as np
import scipy.stats as stats
import pandas as pd
import numpy.testing as npt
import gamma_poisson as gp

class Test_Gamma_Poisson:
    @classmethod
    def setup_class(self):
        """
        Make some sample Markov chain data to run inference on.
        """

        np.random.seed(12346)

        self._setup_constants()

        self._make_transition_probs()

        self._make_initial_states()

        self._make_chain()

        self._make_firing_rates()

        self._make_regressors()

        self._make_count_frame()

    @classmethod
    def _setup_constants(self):
        """
        Set variables that determine the problem size.
        """
        self.U = 100  # units
        self.T = 500  # time points
        self.K = 5  # categories
        self.dt = 1. / 30  # time length of observation (in s)
        self.J = 5  # number of external regressors

    @classmethod
    def _make_transition_probs(self):
        """
        Make a vector of transition probabilities for each category:
        K rows x 2 columns
        Column 0 contains 0 -> 1 transition probability
        Column 1 contains 1 -> 1 transition probability
        """ 
        # want 0 -> to be rare, 1 -> 1 to be a little less so
        v0 = stats.beta.rvs(1, 200, size=self.K)
        v1 = stats.beta.rvs(100, 1, size=self.K)
        self.trans_probs = np.vstack([v0, v1]).T 
        assert_equals(self.trans_probs.shape, (self.K, 2))

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
    def _make_regressors(self):
        """
        make a set of fake binary regressors to be supplied to the model
        """
        X = np.zeros((self.T, self.J), dtype='int')

        # make transition probabilities
        # want 0 -> 1 to be rare, 1 -> 1 to be a little less so
        v0 = stats.beta.rvs(1, 50, size=self.J)
        v1 = stats.beta.rvs(50, 1, size=self.J)
        tp = np.vstack([v0, v1]).T  

        # initialize all to 1
        X[0, :] = stats.bernoulli.rvs(0.5, size=self.J)
        for t in xrange(1, self.T):
            p_trans_to_1 = tp[range(self.J), X[t - 1, :]]
            X[t] = stats.bernoulli.rvs(p_trans_to_1)

        # make data frame of regressors
        Xf = pd.DataFrame(X, columns=map(lambda x: 'X' + str(x), xrange(self.J)))
        Xf.index.name = 'frame'
        Xf = Xf.reset_index()

        self.X = Xf

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

        # add overdispersion
        ss = 10
        rr = 10
        theta = stats.gamma.rvs(a=ss, scale=1./rr, size=fr.shape)
        self.fr *= theta

    @classmethod
    def _make_count_frame(self):
        # draw from Poisson
        N = stats.poisson.rvs(self.fr)

        # convert to dataframe
        df = pd.DataFrame(N)

        # name index frame and turn into column
        df.index.name = 'frame'
        df = df.reset_index()

        # make each frame a row
        df = pd.melt(df, id_vars='frame')

        # set column names
        df.columns = ['frame', 'unit', 'count']

        # pretend all frames from same movie
        df['movie'] = 1

        # concatenate regressors as columns
        df = df.merge(self.X)

        self.N = df

    def test_can_instantiate_model_object(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        assert_is_instance(gpm, gp.GPModel)
        assert_equals(gpm.U, self.U)
        assert_equals(gpm.T, self.T)
        assert_equals(gpm.K, self.K)
        assert_equals(gpm.dt, self.dt)
        assert_equals(gpm.Xframe.shape, 
            (self.T * self.U, self.X.shape[1] - 1))

        # get regressors for first unit
        first_unit = gpm.Xframe.groupby(self.N['frame']).first()
        # compare to X itself
        npt.assert_array_equal(first_unit, self.X.iloc[:, 1:].values)

        assert_is_instance(gpm.Lvalues, list)

    def test_can_handle_no_regressors(self):
        # first, remove extra regressors 
        cols = ['frame', 'unit', 'count', 'movie']
        N = self.N[cols]
        gpm = gp.GPModel(N, self.K, self.dt)
        assert_equals(gpm.J, 0)
        assert_true(not gpm.regressors)
        assert_not_in('aa', gpm.variational_pars)
        assert_not_in('bb', gpm.variational_pars)
        assert_not_in('vv', gpm.prior_pars)
        assert_not_in('ww', gpm.prior_pars)

    def test_can_use_exact_optimization(self):
        gpm = gp.GPModel(self.N, self.K, self.dt, regression_updater='exact')
        assert_equals(gpm.updater, 'exact')

    def test_can_set_priors(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        cctest = np.random.rand(self.K)
        ddtest = np.random.rand(self.K)
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
        gpm = gp.GPModel(self.N, self.K, self.dt)
        gpm.set_priors()
        assert_equals(gpm.cc.shape, (self.K,))
        assert_equals(gpm.dd.shape, (self.K,))
        assert_equals(gpm.nu1.shape, (2, self.K))
        assert_equals(gpm.nu2.shape, (2, self.K))
        assert_equals(gpm.rho1.shape, (self.K,))
        assert_equals(gpm.rho2.shape, (self.K,))

    def test_can_include_overdispersion(self):
        gpm = gp.GPModel(self.N, self.K, self.dt, overdispersion=True)
        gpm.set_priors()
        assert_in('rr', gpm.prior_pars)
        assert_in('ss', gpm.prior_pars)
        assert_equals(gpm.rr.shape, (gpm.U,))
        assert_equals(gpm.ss.shape, (gpm.U,))

        gpm.set_inits()
        assert_in('omega', gpm.variational_pars)
        assert_in('zeta', gpm.variational_pars)
        assert_equals(gpm.omega.shape, (gpm.M,))
        assert_equals(gpm.zeta.shape, (gpm.M,))

    def test_invalid_prior_shapes_raises_error(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
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
        gpm = gp.GPModel(self.N, self.K, self.dt)
        alphatest = np.random.rand(self.K, self.U)
        betatest = np.random.rand(self.K, self.U)
        gamma1test = np.random.rand(2, self.K)
        gamma2test = np.random.rand(2, self.K)
        delta1test = np.random.rand(self.K)
        delta2test = np.random.rand(self.K)
        xitest = np.random.rand(self.T, self.K)
        Xitest = np.random.rand(self.T  - 1, self.K, 2, 2)

        gpm.set_inits(alpha=alphatest, beta=betatest, 
            gamma1=gamma1test, gamma2=gamma2test, delta1=delta1test,
            delta2=delta2test, xi=xitest, Xi=Xitest)
        npt.assert_array_equal(gpm.alpha, alphatest)
        npt.assert_array_equal(gpm.beta, betatest)
        npt.assert_array_equal(gpm.gamma1, gamma1test)
        npt.assert_array_equal(gpm.gamma2, gamma2test)
        npt.assert_array_equal(gpm.delta1, delta1test)
        npt.assert_array_equal(gpm.delta2, delta2test)
        npt.assert_array_equal(gpm.xi, xitest)

        # Xi will have been normalized, and since baseline category is 
        # included by default, Xi[:, k] = [[0, 0] , [0, 1]]
        Xinorm = Xitest / np.sum(Xitest, axis=(-1, -2), keepdims=True)
        npt.assert_allclose(gpm.Xi, Xinorm)
        npt.assert_allclose(np.sum(gpm.Xi, axis=(-1, -2)), 
            np.ones((self.T - 1, self.K)))

        assert_equals(gpm._Ftu.shape, (gpm.T, gpm.U))
        assert_equals(gpm._Ftku.shape, (gpm.T, gpm.K, gpm.U))

    def test_can_set_inits_no_baseline(self):
        gpm = gp.GPModel(self.N, self.K, self.dt, include_baseline=False)
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

        gpm.set_inits(alpha=alphatest, beta=betatest, 
            gamma1=gamma1test, gamma2=gamma2test, delta1=delta1test,
            delta2=delta2test, xi=xitest, Xi=Xitest)
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
        gpm = gp.GPModel(self.N, self.K, self.dt)
        gpm.set_inits()
        assert_equals(gpm.alpha.shape, (self.K, self.U))
        assert_equals(gpm.beta.shape, (self.K, self.U))
        assert_equals(gpm.gamma1.shape, (2, self.K))
        assert_equals(gpm.gamma2.shape, (2, self.K))
        assert_equals(gpm.delta1.shape, (self.K,))
        assert_equals(gpm.delta2.shape, (self.K,))
        assert_equals(gpm.aa.shape, (self.J, self.U))
        assert_equals(gpm.bb.shape, (self.J, self.U))

    def test_invalid_init_shapes_raises_error(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
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

    def test_static_F_prod(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        z = np.random.rand(5, 8)
        w = np.random.rand(8, 3)
        assert_equals(gpm._F_prod(z, w).shape, (5, 8, 3))
        npt.assert_allclose(gpm._F_prod(z, w), 
            np.exp(gpm._F_prod(z, w, log=True)))

        # test whether we can get the same entry in every element by
        # multiplying through by the one part each lacks
        unnorm = 1 - z[..., np.newaxis] + z[..., np.newaxis] * w
        allprod = unnorm * gpm._F_prod(z, w)
        npt.assert_allclose(allprod[0, 0, 0], allprod[0, 1, 0])
        npt.assert_allclose(allprod[-1, 0, 2], allprod[-1, -1, 2])
        npt.assert_allclose(allprod[:, 0, :], gpm._F_prod(z, w, exclude=False))

        # test whether scaling up and w[k] has no effect on _F_prod[:, k]
        k = 1
        wscaled = w.copy()
        wscaled[k] *= 10
        npt.assert_allclose(gpm._F_prod(z, w)[:, k], 
            gpm._F_prod(z, wscaled)[:, k])

    def test_cached_F_prod(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        gpm.set_priors().set_inits()
        gpm.iterate()
        npt.assert_allclose(gpm._Ftu, gpm.F_prod())
        kk = 3
        npt.assert_allclose(gpm._Ftku[:, kk, :], gpm.F_prod(kk))
        npt.assert_allclose(gpm.F_prod(kk, update=True), gpm._Ftku[:, kk, :])
        npt.assert_allclose(np.prod(gpm._Fpre, axis=1), gpm._Ftu)

    def test_cached_G_prod(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        gpm.set_priors().set_inits()
        gpm.iterate()
        npt.assert_allclose(gpm._Gtu, gpm.G_prod())
        kk = 3
        npt.assert_allclose(gpm._Gtku[:, kk], gpm.G_prod(kk))
        npt.assert_allclose(gpm.G_prod(kk, update=True), gpm._Gtku[:, kk])
        npt.assert_allclose(np.prod(gpm._Gpre, axis=1), gpm._Gtu)
        assert_equals(gpm.G_prod().shape[0], gpm.Xframe.shape[0])
        assert_equals(gpm.G_prod(kk).shape[0], gpm.Xframe.shape[0])

    def test_calc_A(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        gpm.set_inits()
        A = gpm.calc_log_A()
        assert_equals(A.shape, (2, 2, self.K))

    def test_L(self):
        # just test that we can run the function without errors
        gpm = gp.GPModel(self.N, self.K, self.dt)
        gpm.set_priors().set_inits()
        L0 = gpm.L()
        assert_equals(L0.shape, ())

    def test_iteration_increases_L(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        gpm.set_priors().set_inits()
        gpm.iterate()  # need to do this to handle wonky initial conditions
        Lvals = [gpm.L()]
        niter = 1 
        for _ in xrange(niter):
            gpm.iterate(keeplog=True)
            Lvals.append(gpm.L())

        # because of inexact solutions to update equations, there may be small
        # decreases in the objective function; we just want to test that these
        # aren't large, which would signal a problem
        change = np.diff(Lvals)
        percent_change = change / np.abs(Lvals[:-1])
        decreases = percent_change[percent_change < 0]
        assert_true(np.all(np.abs(decreases) <= 0.01))

    def test_HMM_entropy_positive(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        gpm.set_priors().set_inits()
        gpm.iterate()  # need to do this to handle wonky initial conditions

        niter = 5 
        for _ in xrange(niter):
            gpm.iterate(keeplog=True)

        Hvals = np.array(gpm.log['H'])
        assert_true(np.all((Hvals > 0) | np.isclose(Hvals, 0)))

    def test_inference_appends_to_Lvalues(self):
        gpm = gp.GPModel(self.N, self.K, self.dt)
        assert_equals(len(gpm.Lvalues), 0)
        gpm.set_priors().set_inits()
        gpm.do_inference(tol=1)
        assert_equals(len(gpm.Lvalues), 1)

    def test_can_optimize_b(self):
        gpm = gp.GPModel(self.N, self.K, self.dt, regression_updater='approximate')
        gpm.set_priors().set_inits()
        gpm.iterate()
        assert_true(~np.isnan(gpm.L()))
        gpm = gp.GPModel(self.N, self.K, self.dt, regression_updater='exact')
        gpm.set_priors().set_inits()
        gpm.iterate()
        assert_true(~np.isnan(gpm.L()))

