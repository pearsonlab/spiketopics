"""
Tests for Gamma model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace
import numpy as np
import scipy.stats as stats
import pandas as pd
import numpy.testing as npt
import gamma_model as gp
from helpers import frames_to_times
import spiketopics.nodes as nd

class Test_Gamma_Model:
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

        self._normalize_count_frame()

    @classmethod
    def _setup_constants(self):
        """
        Set variables that determine the problem size.
        """
        self.U = 100  # units
        self.T = 500  # time points
        self.K = 5  # categories
        self.dt = 1. / 30  # time length of observation (in s)
        self.R = 5  # number of external regressors

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
        X = np.zeros((self.T, self.R), dtype='int')

        # make transition probabilities
        # want 0 -> 1 to be rare, 1 -> 1 to be a little less so
        v0 = stats.beta.rvs(1, 50, size=self.R)
        v1 = stats.beta.rvs(50, 1, size=self.R)
        tp = np.vstack([v0, v1]).T  

        # initialize all to 1
        X[0, :] = stats.bernoulli.rvs(0.5, size=self.R)
        for t in xrange(1, self.T):
            p_trans_to_1 = tp[range(self.R), X[t - 1, :]]
            X[t] = stats.bernoulli.rvs(p_trans_to_1)

        # make data frame of regressors
        Xf = pd.DataFrame(X, columns=map(lambda x: 'X' + str(x), xrange(self.R)))
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
        self.M = self.N.shape[0]

    @classmethod
    def _normalize_count_frame(self):
        self.N = frames_to_times(self.N)


    def test_can_instantiate_model_object(self):
        gpm = gp.GammaModel(self.N, self.K)
        assert_is_instance(gpm, gp.GammaModel)
        assert_equals(gpm.U, self.U)
        assert_equals(gpm.T, self.T)
        assert_equals(gpm.K, self.K)
        assert_equals(gpm.Xframe.shape, 
            (self.T * self.U, self.X.shape[1] - 1))

        # get regressors for first unit
        first_unit = gpm.Xframe.groupby(self.N['time']).first()
        # compare to X itself
        npt.assert_array_equal(first_unit, self.X.iloc[:, 1:].values)

        assert_is_instance(gpm.nodes, dict)

    def test_can_initialize_baseline(self):
        gpm = gp.GammaModel(self.N, self.K)
        prs = np.ones((self.U,))
        prr = np.ones((self.U,))
        pos = np.ones((self.U,))
        por = np.ones((self.U,))
        gpm.initialize_baseline(prior_shape=prs, 
            prior_rate=prr, post_shape=pos, post_rate=por)
        assert_in('baseline', gpm.nodes)
        assert_is_instance(gpm.nodes['baseline'], nd.GammaNode)
        baseline = gpm.nodes['baseline']
        npt.assert_array_equal(prs, baseline.prior_shape)
        npt.assert_array_equal(prr, baseline.prior_rate)
        npt.assert_array_equal(pos, baseline.post_shape)
        npt.assert_array_equal(por, baseline.post_rate)

    def test_can_initialize_baseline_hierarchy(self):
        parent_shape = (1,)
        child_shape = (self.U,)
        ps = np.ones(parent_shape)
        cs = np.ones(child_shape)
        vals = ({'prior_shape_shape': ps, 'prior_shape_rate': ps, 
            'prior_mean_shape': ps, 'prior_mean_rate': ps,
            'post_shape_shape': ps, 'post_shape_rate': ps,
            'post_mean_shape': ps, 'post_mean_rate': ps,
            'post_child_shape': cs, 'post_child_rate': cs})
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**vals)
        assert_in('baseline', gpm.nodes)
        assert_is_instance(gpm.nodes['baseline_shape'], nd.GammaNode)

    def test_can_initialize_fr_latents(self):
        vv = np.random.rand(self.K, self.U)
        vals = ({'prior_shape': vv, 'prior_rate': vv, 
            'post_shape': vv, 'post_rate': vv }) 
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_latents(**vals)
        assert_in('fr_latents', gpm.nodes)
        assert_is_instance(gpm.nodes['fr_latents'], nd.GammaNode)

    def test_can_initialize_fr_latents_hierarchy(self):
        parent_shape = (self.K,)
        child_shape = (self.K, self.U)
        ps = np.random.rand(*parent_shape)
        cs = np.random.rand(*child_shape)
        vals = ({'prior_shape_shape': ps, 'prior_shape_rate': ps, 
            'prior_mean_shape': ps, 'prior_mean_rate': ps,
            'post_shape_shape': ps, 'post_shape_rate': ps,
            'post_mean_shape': ps, 'post_mean_rate': ps,
            'post_child_shape': cs, 'post_child_rate': cs})
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_latents(**vals)
        assert_in('fr_latents', gpm.nodes)
        assert_is_instance(gpm.nodes['fr_latents'], nd.GammaNode)
        assert_is_instance(gpm.nodes['fr_latents_shape'], nd.GammaNode)

    def test_can_initialize_fr_regressors(self):
        vv = np.random.rand(self.R, self.U)
        vals = ({'prior_shape': vv, 'prior_rate': vv, 
            'post_shape': vv, 'post_rate': vv }) 
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_regressors(**vals)
        assert_in('fr_regressors', gpm.nodes)
        assert_is_instance(gpm.nodes['fr_regressors'], nd.GammaNode)

    def test_can_initialize_fr_regressors_hierarchy(self):
        parent_shape = (self.R,)
        child_shape = (self.R, self.U)
        ps = np.random.rand(*parent_shape)
        cs = np.random.rand(*child_shape)
        vals = ({'prior_shape_shape': ps, 'prior_shape_rate': ps, 
            'prior_mean_shape': ps, 'prior_mean_rate': ps,
            'post_shape_shape': ps, 'post_shape_rate': ps,
            'post_mean_shape': ps, 'post_mean_rate': ps,
            'post_child_shape': cs, 'post_child_rate': cs})
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_regressors(**vals)
        assert_in('fr_regressors', gpm.nodes)
        assert_is_instance(gpm.nodes['fr_regressors'], nd.GammaNode)
        assert_is_instance(gpm.nodes['fr_regressors_shape'], nd.GammaNode)

    def test_can_initialize_overdispersion(self):
        vv = np.random.rand(self.M)
        vals = ({'prior_shape': vv, 'prior_rate': vv, 
            'post_shape': vv, 'post_rate': vv }) 
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_overdispersion(**vals)
        assert_in('overdispersion', gpm.nodes)
        assert_is_instance(gpm.nodes['overdispersion'], nd.GammaNode)

    def test_can_initialize_overdispersion_hierarchy(self):
        parent_shape = (1,)
        child_shape = (self.M,)
        ps = np.random.rand(*parent_shape)
        cs = np.random.rand(*child_shape)
        vals = ({'prior_shape_shape': ps, 'prior_shape_rate': ps, 
            'prior_mean_shape': ps, 'prior_mean_rate': ps,
            'post_shape_shape': ps, 'post_shape_rate': ps,
            'post_mean_shape': ps, 'post_mean_rate': ps,
            'post_child_shape': cs, 'post_child_rate': cs})
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_overdispersion(**vals)
        assert_in('overdispersion', gpm.nodes)
        assert_is_instance(gpm.nodes['overdispersion'], nd.GammaNode)
        assert_is_instance(gpm.nodes['overdispersion_shape'], nd.GammaNode)

    def test_can_initialize_latents(self):
        K = self.K
        M = 2
        T = self.T
        A_shape = (M, M, K)
        pi_shape = (M, K)
        z_shape = (M, T, K)
        zz_shape = (M, M, T - 1, K)
        logZ_shape = (K,)
        A = np.ones(A_shape)
        pi = np.ones(pi_shape)
        z = np.ones(z_shape)
        zz = np.ones(zz_shape)
        logZ = np.ones(logZ_shape)
        vals = ({'A_prior': A, 'A_post': A, 'pi_prior': pi, 'pi_post': pi, 
            'z_prior': z, 'zz_prior': zz, 'logZ_prior': logZ})

        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_latents(**vals)

        assert_in('HMM', gpm.nodes)

    def test_F_prod(self):
        # initialize model
        gpm = gp.GammaModel(self.N, self.K)

        # initialize fr effects
        vv = np.random.rand(self.K, self.U)
        ww = np.random.rand(self.K, self.U)
        vals = ({'prior_shape': vv, 'prior_rate': ww, 
            'post_shape': vv, 'post_rate': vv }) 
        gpm.initialize_fr_latents(**vals)

        # initialize latents
        K = self.K
        M = 2
        T = self.T
        A_shape = (M, M, K)
        pi_shape = (M, K)
        z_shape = (M, T, K)
        zz_shape = (M, M, T - 1, K)
        logZ_shape = (K,)
        A = np.random.rand(*A_shape)
        pi = np.random.rand(*pi_shape)
        z = np.random.rand(*z_shape)
        zz = np.random.rand(*zz_shape)
        logZ = np.random.rand(*logZ_shape)
        vals = ({'A_prior': A, 'A_post': A, 'pi_prior': pi, 'pi_post': pi, 
            'z_prior': z, 'zz_prior': zz, 'logZ_prior': logZ})
        gpm.initialize_latents(**vals)

        # initialize/cache F_prod
        gpm.F_prod(update=True)

        # test shapes
        assert_equals(gpm.F_prod(1).shape, (self.T, self.U))
        assert_equals(gpm.F_prod().shape, (self.T, self.U))
        assert_equals(gpm.F_prod(flat=True).shape, (self.M,))

        # test caching
        npt.assert_allclose(gpm.F_prod(1), gpm._Ftku[:, 1, :])
        npt.assert_allclose(gpm.F_prod(), gpm._Ftu)
        npt.assert_allclose(gpm.F_prod(flat=True), gpm._Fflat)

    def test_G_prod(self):
        # initialize model
        gpm = gp.GammaModel(self.N, self.K)

        # initialize fr effects
        vv = np.random.rand(self.K, self.U)
        ww = np.random.rand(self.K, self.U)
        vals = ({'prior_shape': vv, 'prior_rate': ww, 
            'post_shape': vv, 'post_rate': vv }) 
        gpm.initialize_fr_regressors(**vals)

        # initialize/cache F_prod
        gpm.G_prod(update=True)

        # test shapes
        assert_equals(gpm.G_prod(1).shape, (self.M,))
        assert_equals(gpm.G_prod().shape, (self.M,))

        # test caching
        npt.assert_allclose(gpm.G_prod(1), gpm._Gtku[:, 1])
        npt.assert_allclose(gpm.G_prod(), gpm._Gtu)
