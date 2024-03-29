"""
Tests for Gamma model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace, nottest
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

    @classmethod
    def _setup_baseline(self):
        prs = np.ones((self.U,))
        prr = np.ones((self.U,))
        pos = np.ones((self.U,))
        por = np.ones((self.U,))
        self.baseline_dict = ({'prior_shape': prs, 
            'prior_rate': prr, 'post_shape': pos, 
            'post_rate': por})

        parent_shape = ()
        child_shape = (self.U,)
        ps = np.ones(parent_shape)
        cs = np.ones(child_shape)
        self.baseline_hier_dict = ({
            'prior_shape_shape': ps, 'prior_shape_rate': ps, 
            'prior_mean_shape': ps, 'prior_mean_rate': ps,
            'post_shape_shape': ps, 'post_shape_rate': ps,
            'post_mean_shape': ps, 'post_mean_rate': ps,
            'post_child_shape': cs, 'post_child_rate': cs})

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
        'prior_shape': np.random.rand(self.U, self.K),
        'prior_rate': np.random.rand(self.U, self.K),
        'post_shape': np.random.rand(self.U, self.K),
        'post_rate': np.random.rand(self.U, self.K),
        }) 

        parent_shape = (self.K,)
        child_shape = (self.U, self.K)
        ps = np.random.rand(*parent_shape)
        cs = np.random.rand(*child_shape)
        self.fr_latent_hier_dict = ({
            'prior_shape_shape': ps, 'prior_shape_rate': ps, 
            'prior_mean_shape': ps, 'prior_mean_rate': ps,
            'post_shape_shape': ps, 'post_shape_rate': ps,
            'post_mean_shape': ps, 'post_mean_rate': ps,
            'post_child_shape': cs, 'post_child_rate': cs})

    @classmethod
    def _setup_fr_regressors(self):
        self.fr_regressors_dict = ({
        'prior_shape': np.random.rand(self.U, self.R),
        'prior_rate': np.random.rand(self.U, self.R),
        'post_shape': np.random.rand(self.U, self.R),
        'post_rate': np.random.rand(self.U, self.R),
        }) 

        parent_shape = (self.R,)
        child_shape = (self.U, self.R)
        ps = np.random.rand(*parent_shape)
        cs = np.random.rand(*child_shape)
        self.fr_regressors_hier_dict = ({
            'prior_shape_shape': ps, 'prior_shape_rate': ps, 
            'prior_mean_shape': ps, 'prior_mean_rate': ps,
            'post_shape_shape': ps, 'post_shape_rate': ps,
            'post_mean_shape': ps, 'post_mean_rate': ps,
            'post_child_shape': cs, 'post_child_rate': cs})

    @classmethod
    def _setup_overdispersion(self):
        vv = np.random.rand(self.M)
        ww = np.random.rand(self.M)
        self.overdisp_dict = ({'prior_shape': vv, 'prior_rate': vv, 
            'post_shape': ww, 'post_rate': ww }) 

        parent_shape = ()
        child_shape = (self.M,)
        ps = np.random.rand(*parent_shape)
        cs = np.random.rand(*child_shape)
        self.overdisp_hier_dict = ({
            'prior_shape_shape': ps, 'prior_shape_rate': ps, 
            'prior_mean_shape': ps, 'prior_mean_rate': ps,
            'post_shape_shape': ps, 'post_shape_rate': ps,
            'post_mean_shape': ps, 'post_mean_rate': ps,
            'post_child_shape': cs, 'post_child_rate': cs})

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

    def test_negative_X_raises_error(self):
        Nmod = self.N.copy()
        Nmod.loc[0, 'X0'] = -1
        assert_raises(ValueError, gp.GammaModel, Nmod, self.K)

    def test_can_initialize_baseline(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_dict)
        assert_in('baseline', gpm.nodes)
        assert_is_instance(gpm.nodes['baseline'], nd.GammaNode)
        baseline = gpm.nodes['baseline']
        prs = self.baseline_dict['prior_shape']
        prr = self.baseline_dict['prior_rate']
        pos = self.baseline_dict['post_shape']
        por = self.baseline_dict['post_rate']
        npt.assert_array_equal(prs, baseline.prior_shape)
        npt.assert_array_equal(prr, baseline.prior_rate)
        npt.assert_array_equal(pos, baseline.post_shape)
        npt.assert_array_equal(por, baseline.post_rate)

    def test_can_initialize_baseline_hierarchy(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_hier_dict)
        assert_in('baseline', gpm.nodes)
        assert_is_instance(gpm.nodes['baseline_shape'], nd.GammaNode)

    def test_can_initialize_fr_latents(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        assert_in('fr_latents', gpm.nodes)
        assert_is_instance(gpm.nodes['fr_latents'], nd.GammaNode)

    def test_can_initialize_fr_latents_hierarchy(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_latents(**self.fr_latent_hier_dict)
        assert_in('fr_latents', gpm.nodes)
        assert_is_instance(gpm.nodes['fr_latents'], nd.GammaNode)
        assert_is_instance(gpm.nodes['fr_latents_shape'], nd.GammaNode)

    def test_can_initialize_fr_regressors(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        assert_in('fr_regressors', gpm.nodes)
        assert_is_instance(gpm.nodes['fr_regressors'], nd.GammaNode)

    def test_can_initialize_fr_regressors_hierarchy(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_regressors(**self.fr_regressors_hier_dict)
        assert_in('fr_regressors', gpm.nodes)
        assert_is_instance(gpm.nodes['fr_regressors'], nd.GammaNode)
        assert_is_instance(gpm.nodes['fr_regressors_shape'], nd.GammaNode)

    def test_can_initialize_overdispersion(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_overdispersion(**self.overdisp_dict)
        assert_in('overdispersion', gpm.nodes)
        assert_is_instance(gpm.nodes['overdispersion'], nd.GammaNode)

    def test_can_initialize_overdispersion_hierarchy(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_overdispersion(**self.overdisp_hier_dict)
        assert_in('overdispersion', gpm.nodes)
        assert_is_instance(gpm.nodes['overdispersion'], nd.GammaNode)
        assert_is_instance(gpm.nodes['overdispersion_shape'], nd.GammaNode)

    def test_can_initialize_latents(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_latents(**self.latent_dict)

        assert_in('HMM', gpm.nodes)

    def test_finalize(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.finalize()

        assert_true(gpm.latents)
        assert_true(not gpm.regressors)
        assert_true(not gpm.overdispersion)
        gpm.F_prod()

        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        gpm.finalize()

        assert_true(gpm.latents)
        assert_true(gpm.regressors)
        assert_true(not gpm.overdispersion)
        gpm.G_prod()

    def test_F_prod(self):
        # initialize model
        gpm = gp.GammaModel(self.N, self.K)

        gpm.initialize_fr_latents(**self.fr_latent_dict)

        gpm.initialize_latents(**self.latent_dict)

        gpm.finalize()

        # initialize/cache F_prod
        gpm.F_prod(update=True)

        # test shapes
        assert_equals(gpm.F_prod(1).shape, (self.M,))
        assert_equals(gpm.F_prod().shape, (self.M,))

        # test caching
        npt.assert_allclose(gpm.F_prod(1), gpm._Fk[..., 1])
        npt.assert_allclose(gpm.F_prod(), gpm._F)

    def test_G_prod(self):
        # initialize model
        gpm = gp.GammaModel(self.N, self.K)

        # initialize fr effects
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)

        gpm.finalize()

        # initialize/cache F_prod
        gpm.G_prod(update=True)

        # test shapes
        assert_equals(gpm.G_prod(1).shape, (self.M,))
        assert_equals(gpm.G_prod().shape, (self.M,))

        # test caching
        npt.assert_allclose(gpm.G_prod(1), gpm._Gk[..., 1])
        npt.assert_allclose(gpm.G_prod(), gpm._G)

    def test_updates(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_dict)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        gpm.finalize()
        gpm.maxiter = 2
        assert_true(gpm.baseline)
        assert_true(gpm.latents)
        assert_true(gpm.regressors)

        baseline = gpm.nodes['baseline']
        baseline.update()
        uu = gpm.Nframe['unit']
        nn = gpm.Nframe['count']
        npt.assert_allclose(baseline.post_shape, 
            baseline.prior_shape + nn.groupby(uu).sum())

        fr_latents = gpm.nodes['fr_latents']
        fr_latents.update(1)

        fr_regressors = gpm.nodes['fr_regressors']
        fr_regressors.update()

        # add overdispersion
        gpm.initialize_overdispersion(**self.overdisp_dict)
        gpm.finalize()
        od = gpm.nodes['overdispersion']
        od.update()

    def test_hier_updates(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_hier_dict)
        gpm.initialize_fr_latents(**self.fr_latent_hier_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.initialize_fr_regressors(**self.fr_regressors_hier_dict)
        gpm.finalize()

        assert_true(gpm.baseline)
        assert_true(gpm.latents)
        assert_true(gpm.regressors)

        baseline = gpm.nodes['baseline']
        baseline.update()

        fr_latents = gpm.nodes['fr_latents']
        fr_latents.update(1)

        fr_regressors = gpm.nodes['fr_regressors']
        fr_regressors.update()

    def test_calc_log_evidence(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_dict)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        gpm.finalize()

        logpsi = gpm.calc_log_evidence(2)
        assert_equals(logpsi.shape, (self.T, 2))

    def test_expected_log_evidence(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_dict)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        gpm.finalize()

        Elogp = gpm.expected_log_evidence()
        assert_is_instance(Elogp, np.float64)

    def test_duration_dist_optimization(self):
        D = 50
        # add duration dist
        d_hypers = (2.5, 4., 2., 40.)
        d_pars = ({'d_prior_mean': d_hypers[0] * np.ones((2, self.K)), 
          'd_prior_scaling': d_hypers[1] * np.ones((2, self.K)),
          'd_prior_shape': d_hypers[2] * np.ones((2, self.K)),
          'd_prior_rate': d_hypers[3] * np.ones((2, self.K))})
        self.latent_dict.update(d_pars)

        d_inits = (3., 1.1, 1.7, 2.)
        d_post_pars = ({'d_post_mean': d_inits[0] * np.ones((2, self.K)), 
                        'd_post_scaling': d_inits[1] * np.ones((2, self.K)),
                        'd_post_shape': d_inits[2] * np.ones((2, self.K)),
                        'd_post_rate': d_inits[3] * np.ones((2, self.K))})
        self.latent_dict.update(d_post_pars)

        # instatiate model
        gpm = gp.GammaModel(self.N, self.K, D)
        gpm.initialize_baseline(**self.baseline_dict)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        gpm.finalize()

        dnode = gpm.nodes['HMM'].nodes['d']
        par = dnode.parent

        # check that for C = 0, prior and posterior have same parameters
        k = 1

        # before update, priors and inits are correct
        assert_equals(par.post_mean[0, k], d_inits[0])
        assert_equals(par.post_scaling[0, k], d_inits[1])
        assert_equals(par.post_shape[0, k], d_inits[2])
        assert_equals(par.post_rate[0, k], d_inits[3])

        assert_equals(par.prior_mean[0, k], d_hypers[0])
        assert_equals(par.prior_scaling[0, k], d_hypers[1])
        assert_equals(par.prior_shape[0, k], d_hypers[2])
        assert_equals(par.prior_rate[0, k], d_hypers[3])

        # after update with C = 0, posteriors are priors
        dnode.update(k, 0.0)
        npt.assert_allclose(par.post_mean[..., k], 
            par.prior_mean[..., k], rtol=1e-3)
        npt.assert_allclose(par.post_scaling[..., k], 
            par.prior_scaling[..., k], rtol=1e-3)
        npt.assert_allclose(par.post_shape[..., k], 
            par.prior_shape[..., k], rtol=1e-3)
        npt.assert_allclose(par.post_rate[..., k], 
            par.prior_rate[..., k], rtol=1e-3)


        # now set C and scale to be large; after update, logpd should be
        # the ML estimate, which is just C normalized
        lpd = np.empty((2, D))
        mm = np.array([10, 15])
        ss = np.array([0.5, 0.25])
        for m in xrange(2):
            lpd[m] = stats.lognorm.logpdf(xrange(1, D + 1), 
                scale=mm[m], s=ss[m])
        lpd -= np.logaddexp.reduce(lpd, 1, keepdims=True)  # normalization
        scale_up = 1e5
        bigC = scale_up * np.exp(lpd)
        dnode.update(k, bigC)
        import matplotlib.pyplot as plt
        import seaborn as sns
        from helpers import lognormal_from_hypers
        logm = np.empty(2)
        logstd = np.empty(2)
        for m in xrange(2):
            mu = par.post_mean[m, k]
            lam = par.post_scaling[m, k]
            alpha = par.post_shape[m, k]
            beta = par.post_rate[m, k]
            samples = lognormal_from_hypers(mu, lam, alpha, beta, N=1e6)
            valid = (samples >= 1) & (samples <= 50)
            samples = samples[valid]
            logm[m] = np.mean(np.log(samples))
            logstd[m] = np.std(np.log(samples))
            plt.plot(xrange(1, D + 1), np.exp(lpd[m]))
            sns.kdeplot(samples, gridsize=1e5, clip=(1, 50))
        npt.assert_allclose(np.log(mm), logm, rtol=1e-2)
        npt.assert_allclose(ss, logstd, rtol=1e-2)
        # plt.show()

    def test_L(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_dict)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        gpm.finalize()

        assert_is_instance(gpm.L(), np.float64)
        initial_log_len = len(gpm.log['L'])
        gpm.L(keeplog=True)
        assert_equals(len(gpm.log['L']), initial_log_len + 1)

    def test_iterate(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_hier_dict)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        gpm.finalize()

        L_init = gpm.L(keeplog=True)
        gpm.iterate(keeplog=True, verbosity=2)
        assert_true(gpm.L() > L_init)

    def test_inference(self):
        gpm = gp.GammaModel(self.N, self.K)
        gpm.initialize_baseline(**self.baseline_hier_dict)
        gpm.initialize_fr_latents(**self.fr_latent_dict)
        gpm.initialize_latents(**self.latent_dict)
        gpm.initialize_fr_regressors(**self.fr_regressors_dict)
        gpm.finalize()
        gpm.maxiter = 2

        gpm.iterate()
        assert_true(~np.isnan(gpm.L()))
