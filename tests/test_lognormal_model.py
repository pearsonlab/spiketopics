"""
Tests for lognormal model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace, nottest
import numpy as np
import scipy.stats as stats
import pandas as pd
import numpy.testing as npt
import lognormal_model as ln
from helpers import frames_to_times
import spiketopics.nodes as nd

class Test_LogNormal_Model:
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

        self._setup_baseline()

        self._setup_latents()

        self._setup_fr_latents()

        self._setup_fr_regressors()

        self._setup_overdispersion()


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
        bl = stats.lognorm.rvs(scale=10, s=0.5, size=self.U)
        lam = stats.lognorm.rvs(s=0.25, size=(self.K, self.U))

        # calculate rate for each time
        fr = bl * np.exp(self.chain.T.dot(np.log(lam))) * self.dt
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

    @classmethod
    def _setup_baseline(self):
        prs = np.array(1)
        prr = np.array(1)
        pos = np.ones((self.U,))
        por = np.ones((self.U,))
        self.baseline_dict = ({'prior_mean': prs, 
            'prior_prec': prr, 'post_mean': pos, 
            'post_prec': por})

    @classmethod
    def _setup_latents(self):
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
        self.latent_dict = ({'A_prior': A, 'A_post': A, 'pi_prior': pi, 'pi_post': pi, 'z_init': z, 'zz_init': zz, 'logZ_init': logZ})

    @classmethod
    def _setup_fr_latents(self):
        self.fr_latent_dict = ({
        'prior_mean': np.random.rand(self.K,),
        'prior_prec': np.random.rand(self.K,),
        'post_mean': np.random.rand(self.U, self.K),
        'post_prec': np.random.rand(self.U, self.K),
        }) 

        grandparent_shape = ()
        parent_shape = (self.K,)
        child_shape = (self.U, self.K)
        gs = np.random.rand(*grandparent_shape)
        ps = np.random.rand(*parent_shape)
        cs = np.random.rand(*child_shape)
        self.fr_latent_hier_dict = ({
            'prior_prec_shape': gs, 
            'prior_prec_rate': gs, 
            'post_mean': cs,
            'post_prec': cs,
            'post_prec_shape': ps,
            'post_prec_rate': ps
            })

    @classmethod
    def _setup_fr_regressors(self):
        self.fr_regressors_dict = ({
        'prior_mean': np.random.rand(self.R,),
        'prior_prec': np.random.rand(self.R,),
        'post_mean': np.random.rand(self.U, self.R),
        'post_prec': np.random.rand(self.U, self.R),
        }) 

        grandparent_shape = ()
        parent_shape = (self.R,)
        child_shape = (self.U, self.R)
        gs = np.random.rand(*grandparent_shape)
        ps = np.random.rand(*parent_shape)
        cs = np.random.rand(*child_shape)
        self.fr_regressors_hier_dict = ({
            'prior_prec_shape': gs, 
            'prior_prec_rate': gs, 
            'post_mean': cs,
            'post_prec': cs,
            'post_prec_shape': ps,
            'post_prec_rate': ps
            })

    @classmethod
    def _setup_overdispersion(self):
        prs = np.array(1)
        prr = np.array(1)
        pos = np.ones((self.M,))
        por = np.ones((self.M,))
        self.overdisp_dict = ({'prior_mean': prs, 
            'prior_prec': prr, 'post_mean': pos, 
            'post_prec': por})

    def test_can_instantiate_model_object(self):
        lnm = ln.LogNormalModel(self.N, self.K)
        assert_is_instance(lnm, ln.LogNormalModel)
        assert_equals(lnm.U, self.U)
        assert_equals(lnm.T, self.T)
        assert_equals(lnm.K, self.K)
        assert_equals(lnm.Xframe.shape, 
            (self.T * self.U, self.X.shape[1] - 1))

        # get regressors for first unit
        first_unit = lnm.Xframe.groupby(self.N['time']).first()
        # compare to X itself
        npt.assert_array_equal(first_unit, self.X.iloc[:, 1:].values)

        assert_is_instance(lnm.nodes, dict)

    def test_can_initialize_baseline(self):
        lnm = ln.LogNormalModel(self.N, self.K)
        lnm.initialize_baseline(**self.baseline_dict)
        assert_in('baseline', lnm.nodes)
        assert_is_instance(lnm.nodes['baseline'], nd.GaussianNode)
        baseline = lnm.nodes['baseline']
        prs = self.baseline_dict['prior_mean']
        prr = self.baseline_dict['prior_prec']
        pos = self.baseline_dict['post_mean']
        por = self.baseline_dict['post_prec']
        npt.assert_array_equal(prs, baseline.prior_mean)
        npt.assert_array_equal(prr, baseline.prior_prec)
        npt.assert_array_equal(pos, baseline.post_mean)
        npt.assert_array_equal(por, baseline.post_prec)

    def test_can_initialize_fr_latents(self):
        lnm = ln.LogNormalModel(self.N, self.K)
        lnm.initialize_fr_latents(**self.fr_latent_dict)
        assert_in('fr_latents', lnm.nodes)
        assert_is_instance(lnm.nodes['fr_latents'], nd.GaussianNode)

    def test_can_initialize_fr_latents_hierarchy(self):
        lnm = ln.LogNormalModel(self.N, self.K)
        lnm.initialize_fr_latents(**self.fr_latent_hier_dict)
        assert_in('fr_latents', lnm.nodes)
        assert_is_instance(lnm.nodes['fr_latents'], nd.GaussianNode)
        assert_is_instance(lnm.nodes['fr_latents_prec'], nd.GammaNode)

    def test_can_initialize_fr_regressors(self):
        lnm = ln.LogNormalModel(self.N, self.K)
        lnm.initialize_fr_regressors(**self.fr_regressors_dict)
        assert_in('fr_regressors', lnm.nodes)
        assert_is_instance(lnm.nodes['fr_regressors'], nd.GaussianNode)

    def test_can_initialize_fr_regressors_hierarchy(self):
        lnm = ln.LogNormalModel(self.N, self.K)
        lnm.initialize_fr_regressors(**self.fr_regressors_hier_dict)
        assert_in('fr_regressors', lnm.nodes)
        assert_is_instance(lnm.nodes['fr_regressors'], nd.GaussianNode)
        assert_is_instance(lnm.nodes['fr_regressors_prec'], nd.GammaNode)

    # def test_can_initialize_overdispersion(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_overdispersion(**self.overdisp_dict)
    #     assert_in('overdispersion', lnm.nodes)
    #     assert_is_instance(lnm.nodes['overdispersion'], nd.GammaNode)

    # def test_can_initialize_overdispersion_hierarchy(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_overdispersion(**self.overdisp_hier_dict)
    #     assert_in('overdispersion', lnm.nodes)
    #     assert_is_instance(lnm.nodes['overdispersion'], nd.GammaNode)
    #     assert_is_instance(lnm.nodes['overdispersion_shape'], nd.GammaNode)

    # def test_can_initialize_latents(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_latents(**self.latent_dict)

    #     assert_in('HMM', lnm.nodes)

    # def test_finalize(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_fr_latents(**self.fr_latent_dict)
    #     lnm.initialize_latents(**self.latent_dict)
    #     lnm.finalize()

    #     assert_true(lnm.latents)
    #     assert_true(not lnm.regressors)
    #     assert_true(not lnm.overdispersion)
    #     lnm.F_prod()

    #     lnm.initialize_fr_regressors(**self.fr_regressors_dict)
    #     lnm.finalize()

    #     assert_true(lnm.latents)
    #     assert_true(lnm.regressors)
    #     assert_true(not lnm.overdispersion)
    #     lnm.G_prod()

    # def test_F_prod(self):
    #     # initialize model
    #     lnm = ln.LogNormalModel(self.N, self.K)

    #     lnm.initialize_fr_latents(**self.fr_latent_dict)

    #     lnm.initialize_latents(**self.latent_dict)

    #     lnm.finalize()

    #     # initialize/cache F_prod
    #     lnm.F_prod(update=True)

    #     # test shapes
    #     assert_equals(lnm.F_prod(1).shape, (self.T, self.U))
    #     assert_equals(lnm.F_prod().shape, (self.T, self.U))
    #     assert_equals(lnm.F_prod(flat=True).shape, (self.M,))

    #     # test caching
    #     npt.assert_allclose(lnm.F_prod(1), lnm._Ftuk[..., 1])
    #     npt.assert_allclose(lnm.F_prod(), lnm._Ftu)
    #     npt.assert_allclose(lnm.F_prod(flat=True), lnm._Ftu_flat)

    # def test_G_prod(self):
    #     # initialize model
    #     lnm = ln.LogNormalModel(self.N, self.K)

    #     # initialize fr effects
    #     lnm.initialize_fr_regressors(**self.fr_regressors_dict)

    #     lnm.finalize()

    #     # initialize/cache F_prod
    #     lnm.G_prod(update=True)

    #     # test shapes
    #     assert_equals(lnm.G_prod(1).shape, (self.T, self.U))
    #     assert_equals(lnm.G_prod().shape, (self.T, self.U))
    #     assert_equals(lnm.G_prod(flat=True).shape, (self.M,))

    #     # test caching
    #     npt.assert_allclose(lnm.G_prod(1), lnm._Gtuk[..., 1])
    #     npt.assert_allclose(lnm.G_prod(), lnm._Gtu)
    #     npt.assert_allclose(lnm.G_prod(flat=True), lnm._Gtu_flat)

    # def test_updates(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_baseline(**self.baseline_dict)
    #     lnm.initialize_fr_latents(**self.fr_latent_dict)
    #     lnm.initialize_latents(**self.latent_dict)
    #     lnm.initialize_fr_regressors(**self.fr_regressors_dict)
    #     lnm.finalize()
    #     assert_true(lnm.baseline)
    #     assert_true(lnm.latents)
    #     assert_true(lnm.regressors)

    #     baseline = lnm.nodes['baseline']
    #     baseline.update()
    #     npt.assert_allclose(baseline.post_shape, 
    #         baseline.prior_shape + np.sum(lnm.N, axis=0))

    #     fr_latents = lnm.nodes['fr_latents']
    #     fr_latents.update(1)

    #     fr_regressors = lnm.nodes['fr_regressors']
    #     fr_regressors.update()

    #     # add overdispersion
    #     lnm.initialize_overdispersion(**self.overdisp_dict)
    #     lnm.finalize()
    #     od = lnm.nodes['overdispersion']
    #     od.update()

    # def test_hier_updates(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_baseline(**self.baseline_hier_dict)
    #     lnm.initialize_fr_latents(**self.fr_latent_hier_dict)
    #     lnm.initialize_latents(**self.latent_dict)
    #     lnm.initialize_fr_regressors(**self.fr_regressors_hier_dict)
    #     lnm.finalize()
    #     assert_true(lnm.baseline)
    #     assert_true(lnm.latents)
    #     assert_true(lnm.regressors)

    #     baseline = lnm.nodes['baseline']
    #     baseline.update()

    #     fr_latents = lnm.nodes['fr_latents']
    #     fr_latents.update(1)

    #     fr_regressors = lnm.nodes['fr_regressors']
    #     fr_regressors.update()

    # def test_calc_log_evidence(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_baseline(**self.baseline_dict)
    #     lnm.initialize_fr_latents(**self.fr_latent_dict)
    #     lnm.initialize_latents(**self.latent_dict)
    #     lnm.initialize_fr_regressors(**self.fr_regressors_dict)
    #     lnm.finalize()

    #     lolnsi = lnm.calc_log_evidence(2)
    #     assert_equals(lolnsi.shape, (self.T, 2))

    # def test_expected_log_evidence(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_baseline(**self.baseline_dict)
    #     lnm.initialize_fr_latents(**self.fr_latent_dict)
    #     lnm.initialize_latents(**self.latent_dict)
    #     lnm.initialize_fr_regressors(**self.fr_regressors_dict)
    #     lnm.finalize()

    #     Eloln = lnm.expected_log_evidence()
    #     assert_is_instance(Eloln, np.float64)

    # def test_L(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_baseline(**self.baseline_dict)
    #     lnm.initialize_fr_latents(**self.fr_latent_dict)
    #     lnm.initialize_latents(**self.latent_dict)
    #     lnm.initialize_fr_regressors(**self.fr_regressors_dict)
    #     lnm.finalize()

    #     assert_is_instance(lnm.L(), np.float64)
    #     initial_log_len = len(lnm.log['L'])
    #     L_init = lnm.L(keeplog=True)
    #     assert_equals(len(lnm.log['L']), initial_log_len + 1)

    # def test_iterate(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_baseline(**self.baseline_hier_dict)
    #     lnm.initialize_fr_latents(**self.fr_latent_dict)
    #     lnm.initialize_latents(**self.latent_dict)
    #     lnm.initialize_fr_regressors(**self.fr_regressors_dict)
    #     lnm.finalize()

    #     L_init = lnm.L(keeplog=True)
    #     lnm.iterate(keeplog=True, verbosity=2)
    #     assert_true(lnm.L() > L_init)

    # def test_inference(self):
    #     lnm = ln.LogNormalModel(self.N, self.K)
    #     lnm.initialize_baseline(**self.baseline_hier_dict)
    #     lnm.initialize_fr_latents(**self.fr_latent_dict)
    #     lnm.initialize_latents(**self.latent_dict)
    #     lnm.initialize_fr_regressors(**self.fr_regressors_dict)
    #     lnm.finalize()

    #     lnm.iterate()
    #     assert_true(~np.isnan(lnm.L()))
