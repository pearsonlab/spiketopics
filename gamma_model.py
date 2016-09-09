from __future__ import division
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numexpr as ne
import spiketopics.nodes as nd
from numba import jit


class GammaModel(object):
    """
    This class fits a Poisson observation model using a product of Gamma-
    distributed variables to model the rates.

    N ~ Poiss(\mu)
    \mu = prod \lambda
    \lambda ~ Gamma
    """

    def __init__(self, data, K, D=None):
        """
        Construct model object.

        data: Pandas DataFrame with one row per observation and columns
            'unit', 'time', and 'count' (time is stimulus time)
        K: number of latent categories to infer
        D: None for HMM; for HSMM, maximum duration
        nodedict: dictionary of nodes in the model; node names can be
            'baseline': baseline firing rates
            'regressor': firing rate effects for each regressor
            'latent': firing rate effects for latent states
            'overdispersion': firing rate effects due to overdispersion
            All of the above nodes are optional. Additional nodes (e.g.,
            parents of the above) are permitted, and will be updated
            appropriately.
        """
        # Infer basic constants
        M = data.shape[0]  # number of observations
        # regressors should be columns other than unit, time, and count
        R = data.shape[1] - 3  # number of regressor columns
        T = data['time'].drop_duplicates().shape[0]  # number unique stim times
        U = data['unit'].drop_duplicates().shape[0]  # number unique units

        self.M = M
        self.R = R
        self.T = T
        self.K = K
        self.U = U
        self.D = D

        self.nodes = {}  # dict for variable nodes in graphical model

        # maximum number of iterations for regressor optimization step
        self.maxiter = 1000

        self.log = {'L': []}  # for debugging
        self.Lvalues = []  # for recording optimization objective each iter

        self._parse_frames(data)

    def _parse_frames(self, data):
        """
        Split input dataframe data into two pieces: one of count observations
        (Nframe), one of regressors (Xframe). Also make two helpful arrays, one
        of spike counts at each time, one of number of observations of each
        time.
        """
        # Modified by Xin
        cols = ['unit', 'time', 'trial', 'count']
        # cols = ['unit', 'time', 'count']
        self.Nframe = data[cols].copy()

        # empty data frame for simple model
        self.Xframe = data.drop(cols, axis=1).copy()
        if self.Xframe.values.size > 0 and np.min(self.Xframe.values) < 0.0:
            raise ValueError('User-specified regressors must be non-negative')

        # make sure units are indexed from 0
        self.Nframe['unit'] = self.Nframe['unit'] - np.min(self.Nframe['unit'])

        return self

    def _initialize_gamma_nodes(self, name, node_shape, parent_shape,
                                **kwargs):
        """
        Perform multiple distpatch for initializing gamma nodes based on
            parameters in kwargs.
        name: name of the base node
        node_shape: shape of base node
        parent_shape: shape of parent node
        """
        # test if parameters indicate there's a hierarchy
        if 'prior_sparsity' in kwargs:
            nodes = nd.initialize_sparse_gamma_hierarchy(name, parent_shape,
                                                         node_shape, **kwargs)
        elif 'prior_shape_shape' in kwargs:
            nodes = nd.initialize_gamma_hierarchy(name, parent_shape,
                                                  node_shape, **kwargs)
        elif 'mapper' in kwargs:
            nodes = nd.initialize_ragged_gamma_hierarchy(name, parent_shape,
                                                         node_shape, **kwargs)
        else:
            nodes = nd.initialize_gamma(name, node_shape, **kwargs)

        for n in nodes:
            self.nodes.update({n.name: n})

    def initialize_baseline(self, **kwargs):
        """
        Set up node for baseline firing rates.
        Assumes the prior is on f * dt, where f is the baseline firing
        rate and dt is the time bin size.
        """
        self._initialize_gamma_nodes('baseline', (self.U,), (), **kwargs)

        return self

    def initialize_fr_latents(self, **kwargs):
        """
        Set up node for firing rate effects due to latent variables.
        """
        self._initialize_gamma_nodes('fr_latents', (self.U, self.K),
                                     (self.K,), **kwargs)

        return self

    def initialize_fr_regressors(self, **kwargs):
        """
        Set up node for firing rate effects due to latent variables.
        """
        self._initialize_gamma_nodes('fr_regressors', (self.U, self.R),
                                     (self.R,), **kwargs)

        return self

    def initialize_overdispersion(self, **kwargs):
        """
        Set up trial-to-trial overdispersion on firing rates.
        """
        self._initialize_gamma_nodes('overdispersion', (self.M,),
                                     (self.U,), **kwargs)

        return self

    def initialize_overdispersion_natural(self, **kwargs):
        """
        Set up trial-to-trial overdispersion natural time on firing rates.
        """
        self._initialize_gamma_nodes('overdispersion_natural', (self.M,),
                                     (self.U,), **kwargs)

        return self

    def initialize_latents(self, **kwargs):
        nodes = nd.initialize_HMM(self.K, 2, self.T, self.D, **kwargs)

        self.nodes.update({n.name: n for n in nodes})

        return self

    def _sort_values(self, indmat, values):
        """
        :param indmat: an n*2 matrix with Column 1: original index, Column 2: sorted index
        :param values: an n vector to be sorted according to column 2
        :return: sorted values
        """
        sorted_values = np.c_[indmat[indmat[:, 0].argsort()], values]
        sorted_values = sorted_values[sorted_values[:, 1].argsort()][:, 2]
        return sorted_values

    def _unsort_values(self, indmat, values):
        """
        :param indmat: an n*2 matrix with Column 1: original index, Column 2: sorted index
        :param values: an n vector to be unsorted according to column 1
        :return: unsorted values
        """
        unsort_values = np.c_[indmat[indmat[:, 1].argsort()], values]
        unsort_values = unsort_values[unsort_values[:, 0].argsort()][:, 2]
        return unsort_values

    def update_baseline(self):
        node = self.nodes['baseline']

        uu = self.Nframe['unit']
        nn = self.Nframe['count']

        if self.overdispersion:
            od = self.nodes['overdispersion'].expected_x()
        elif self.overdispersion_natural:
            # # cumulated products of expectation of phi
            # expected_phi = self.nodes['overdispersion_natural'].expected_x()
            # od = np.exp(np.cumsum(np.log(expected_phi).reshape(-1, 
            #     self.time_natural), axis=1)).ravel()

            # get the dataframe for count values
            cnt_index = self.Nframe.sort_values(['unit', 'trial', 'time']).index
            # index matrix of 2 columns: Column 1 original, Column 2 sorted
            ind_mat = np.c_[np.array(cnt_index),
                            np.array(xrange(cnt_index.shape[0]))]

            # expectation of phi
            expected_phi = self.nodes['overdispersion_natural'].expected_x()

            # sort expected_phi and compute cumulative product
            exphi_sorted = self._sort_values(ind_mat, expected_phi)
            # cumulative product
            exphi_sorted = np.cumprod(exphi_sorted.reshape(-1, self.time_natural), axis=1).ravel()
            # unsort expected_phi
            od = self._unsort_values(ind_mat, exphi_sorted)
        else:
            od = 1

        F = self.F_prod()
        G = self.G_prod()
        allprod = od * F * G
        eff_rate = pd.DataFrame(allprod).groupby(uu).sum().values.squeeze()

        node.post_shape = (node.prior_shape.expected_x() +
                           nn.groupby(uu).sum().values)
        node.post_rate = node.prior_rate.expected_x() + eff_rate

    def update_fr_latents(self, idx):
        uu = self.Nframe['unit']
        tt = self.Nframe['time']
        nn = self.Nframe['count']

        lam = self.nodes['fr_latents']
        xi = self.nodes['HMM'].nodes['z'].z[1, :, idx]
        Nz = (xi[tt] * nn).groupby(uu).sum().values

        if self.overdispersion:
            od = self.nodes['overdispersion'].expected_x()
        elif self.overdispersion_natural:
            # # cumulated products of expectation of phi
            # expected_phi = self.nodes['overdispersion_natural'].expected_x()
            # od = np.exp(np.cumsum(np.log(expected_phi).reshape(-1, 
            #     self.time_natural), axis=1)).ravel()

            # get the dataframe for count values
            cnt_index = self.Nframe.sort_values(['unit', 'trial', 'time']).index
            # index matrix of 2 columns: Column 1 original, Column 2 sorted
            ind_mat = np.c_[np.array(cnt_index),
                            np.array(xrange(cnt_index.shape[0]))]

            # expectation of phi
            expected_phi = self.nodes['overdispersion_natural'].expected_x()

            # sort expected_phi and compute cumulative product
            exphi_sorted = self._sort_values(ind_mat, expected_phi)
            # cumulative product
            exphi_sorted = np.cumprod(exphi_sorted.reshape(-1, self.time_natural), axis=1).ravel()
            # unsort expected_phi
            od = self._unsort_values(ind_mat, exphi_sorted)
        else:
            od = 1

        bl = self.nodes['baseline'].expected_x()[uu]
        Fz = self.F_prod(idx) * xi[tt]
        G = self.G_prod()
        allprod = bl * od * Fz * G
        eff_rate = pd.DataFrame(allprod).groupby(uu).sum().values.squeeze()

        # WARNING:
        # this is super hacky, but getting around it would require
        # writing both a CategoricalNode and a MixtureNode, and I'm not
        # yet convinced it's worth it
        if 'fr_latents_spike' in self.nodes:
            alpha = self.nodes['fr_latents_feature'].expected_x()[..., idx]
            Eslab_shape = lam.prior_shape.expected_x()[..., idx]
            Eslab_rate = lam.prior_rate.expected_x()[..., idx]
            C = self.nodes['fr_latents_spike'].prior_shape
            Eshape = alpha * Eslab_shape + (1 - alpha) * C
            Erate = alpha * Eslab_rate + (1 - alpha) * C
        else:
            Eshape = lam.prior_shape.expected_x()[..., idx]
            Erate = lam.prior_rate.expected_x()[..., idx]

        lam.post_shape[..., idx] = Eshape + Nz
        lam.post_rate[..., idx] = Erate + eff_rate
        self.F_prod(idx, update=True)

    def calc_log_evidence(self, idx):
        """
        Calculate p(N|z, rest) for use in updating HMM. Need only be
        correct up to an overall constant.
        """
        logpsi = np.empty((self.T, 2))
        N = self.Nframe
        nn = N['count']
        uu = N['unit']

        if self.overdispersion:
            od = self.nodes['overdispersion'].expected_x()
        elif self.overdispersion_natural:
            # # cumulated products of phi
            # expected_phi = self.nodes['overdispersion_natural'].expected_x()
            # od = np.exp(np.cumsum(np.log(expected_phi).reshape(-1, 
            #     self.time_natural), axis=1)).ravel()

            # get the dataframe for count values
            cnt_index = self.Nframe.sort_values(['unit', 'trial', 'time']).index
            # index matrix of 2 columns: Column 1 original, Column 2 sorted
            ind_mat = np.c_[np.array(cnt_index),
                            np.array(xrange(cnt_index.shape[0]))]

            # expectation of phi
            expected_phi = self.nodes['overdispersion_natural'].expected_x()

            # sort expected_phi and compute cumulative product
            exphi_sorted = self._sort_values(ind_mat, expected_phi)
            # cumulative product
            exphi_sorted = np.cumprod(exphi_sorted.reshape(-1, self.time_natural), axis=1).ravel()
            # unsort expected_phi
            od = self._unsort_values(ind_mat, exphi_sorted)
        else:
            od = 1
        bl = self.nodes['baseline'].expected_x()[uu]
        Fk = self.F_prod(idx)
        G = self.G_prod()

        allprod = bl * od * Fk * G

        lam = self.nodes['fr_latents']
        bar_log_lam = lam.expected_log_x()[uu, idx]
        bar_lam = lam.expected_x()[uu, idx]

        N['lam0'] = -allprod
        N['lam1'] = -(allprod * bar_lam) + (nn * bar_log_lam)

        logpsi = N.groupby('time').sum()[['lam0', 'lam1']].values

        return logpsi

    @jit
    def expected_log_evidence(self):
        """
        Calculate E[log p(N, z|rest).
        """
        uu = self.Nframe['unit']
        nn = self.Nframe['count']
        tt = self.Nframe['time']

        Elogp = 0
        eff_rate = 1

        if self.baseline:
            node = self.nodes['baseline']
            bar_log_lam = node.expected_log_x()
            bar_lam = node.expected_x()

            Elogp += np.sum(nn * bar_log_lam[uu])
            eff_rate *= bar_lam[uu]

        if self.latents:
            node = self.nodes['fr_latents']
            bar_log_lam = node.expected_log_x()
            xi = self.nodes['HMM'].nodes['z'].z[1]

            Elogp += np.sum(nn[:, np.newaxis] * xi[tt] * bar_log_lam[uu])
            eff_rate *= self.F_prod()

            # pieces for A and pi
            Elogp += self.nodes['HMM'].expected_log_state_sequence()

        if self.regressors:
            node = self.nodes['fr_regressors']
            bar_log_lam = node.expected_log_x()
            xx = self.Xframe.values

            Elogp += np.sum(nn[:, np.newaxis] * xx * bar_log_lam[uu])
            eff_rate *= self.G_prod()

        if self.overdispersion:
            node = self.nodes['overdispersion']
            bar_log_lam = node.expected_log_x()
            bar_lam = node.expected_x()

            Elogp += np.sum(nn * bar_log_lam)
            # Modified by Xin to compare distributions of bar_lam
            # print "Min, Mean, Max of bar_lam: {}\t{}\t{}".format(
            #     np.min(bar_lam), np.mean(bar_lam), np.max(bar_lam))

            eff_rate *= bar_lam
            # print "Min, Mean, Max of eff_rate: {}\t{}\t{}!!!!!!".format(
            #     np.min(eff_rate), np.mean(eff_rate), np.max(eff_rate))

        if self.overdispersion_natural:
            #print "compute overdispersion natural!"
            node = self.nodes['overdispersion_natural']
            bar_log_phi = node.expected_log_x()
            bar_phi = node.expected_x()
            time_nat = self.time_natural

            # cnt_array = self.Nframe.sort_values(['unit', 'trial', 'time'])['count'].values
            # cnt_cumsum = np.cumsum(cnt_array.reshape(-1, time_nat)[:, ::-1], axis=1
            #                        )[:, ::-1].ravel()
            # Elogp += np.sum(cnt_cumsum * bar_log_phi)

            # prod_mat = bar_phi.reshape(-1, time_nat)
            # prod_array = np.exp(np.cumsum(np.log(prod_mat), axis=1)).ravel()
            # # print "\nMin, Mean, Max of product array:  {}\t{}\t{}".format(
            # #     np.min(prod_array), np.mean(prod_array), np.max(prod_array))

            # eff_rate *= prod_array
            # # print "Min, Mean, Max of effective rate: {}\t{}\t{}!!!!!!".format(
            # #     np.min(eff_rate), np.mean(eff_rate), np.max(eff_rate))

            # get the dataframe for count values
            cnt_dframe = self.Nframe.sort_values(['unit', 'trial', 'time'])['count']
            cnt_cumsum = np.cumsum(np.array(cnt_dframe).reshape(-1, time_nat)[:, ::-1], axis=1)[:, ::-1].ravel()

            # index matrix of 2 columns: Column 1 original, Column 2 sorted
            ind_mat = np.c_[np.array(cnt_dframe.index), np.array(xrange(cnt_dframe.index.shape[0]))]
            # unsort cnt_cumsum to be in the original order
            cumsum_unsort = self._unsort_values(ind_mat, cnt_cumsum)

            # cnt_array = self.Nframe.sort_values(['unit', 'trial', 'time'])['count'].values
            # cnt_cumsum = np.cumsum(cnt_array.reshape(-1, time_nat)[:, ::-1], axis=1
            #                       )[:, ::-1].ravel()
            Elogp += np.sum(cumsum_unsort * bar_log_phi)

            # sort bar_phi to compute cumulative product
            exphi_sorted =self. _sort_values(ind_mat, bar_phi)
            # compute the cumultive product
            prod_array = np.cumprod(exphi_sorted.reshape(-1, time_nat), axis=1).ravel()
            print "Min, Mean, Max of product array:  {}\t{}\t{}".format(
                np.min(prod_array), np.mean(prod_array), np.max(prod_array))

            # sort prod_array to be in the original order
            prod_unsort = self._unsort_values(ind_mat, prod_array)

            eff_rate *= prod_unsort
            print "Min, Mean, Max of effective rate: {}\t{}\t{}!!!!!!".format(
                np.min(eff_rate), np.mean(eff_rate), np.max(eff_rate))

        Elogp += -np.sum(eff_rate)

        return Elogp

    def update_fr_regressors(self):
        lam = self.nodes['fr_regressors']
        nn = self.Nframe['count']
        uu = self.Nframe['unit']
        NX = nn[:, np.newaxis] * self.Xframe

        lam.post_shape = (lam.prior_shape.expected_x().reshape(-1, self.R) +
                          NX.groupby(uu).sum().values)

        # now to find the rates, we have to optimize
        starts = lam.post_rate
        lam.post_rate = self.optimize_regressor_rates(starts)
        self.G_prod(update=True)

    def optimize_regressor_rates(self, starts):
        """
        Solve for log(prior_rate) via black-box optimization.
        Use log(b) since this is the natural parameter.
        Updater is the name of a factory function that returns a function
        to be minimized based on current parameter values.
        """
        uu = self.Nframe['unit'].values.astype('int64')
        aa = self.nodes['fr_regressors'].post_shape
        ww = self.nodes['fr_regressors'].prior_rate.expected_x()
        ww = ww.view(np.ndarray).reshape(-1, self.R)

        if self.overdispersion:
            od = self.nodes['overdispersion'].expected_x()
        elif self.overdispersion_natural:
            # # cumulated products of expectation of phi
            # expected_phi = self.nodes['overdispersion_natural'].expected_x()
            # od = np.exp(np.cumsum(np.log(expected_phi).reshape(-1, 
            #     self.time_natural), axis=1)).ravel()

            # get the dataframe for count values
            cnt_index = self.Nframe.sort_values(['unit', 'trial', 'time']).index
            # index matrix of 2 columns: Column 1 original, Column 2 sorted
            ind_mat = np.c_[np.array(cnt_index),
                            np.array(xrange(cnt_index.shape[0]))]

            # expectation of phi
            expected_phi = self.nodes['overdispersion_natural'].expected_x()

            # sort expected_phi and compute cumulative product
            exphi_sorted = self._sort_values(ind_mat, expected_phi)
            # cumulative product
            exphi_sorted = np.cumprod(exphi_sorted.reshape(-1, self.time_natural), axis=1).ravel()
            # unsort expected_phi
            od = self._unsort_values(ind_mat, exphi_sorted)
        else:
            od = 1

        F = self.F_prod()
        bl = self.nodes['baseline'].expected_x()[uu]
        Fblod = np.array(F * bl * od)

        minfun = exact_minfun

        eps_starts = np.log(aa / starts)

        res = minimize(minfun, eps_starts,
                       args=(aa, ww, uu, Fblod, self.Xframe.values),
                       jac=True, options={'maxiter': self.maxiter})

        if not res.success:
            print "Warning: optimization terminated without success."
            print res.message
        eps = res.x.reshape(self.U, self.R)
        bb = aa * np.exp(-eps)
        return bb

    def update_overdispersion(self):
        node = self.nodes['overdispersion']
        nn = self.Nframe['count']
        uu = self.Nframe['unit']
        bl = self.nodes['baseline'].expected_x()[uu]
        F = self.F_prod()
        G = self.G_prod()

        if node.has_parents:
            node.post_shape = node.prior_shape.expected_x()[uu] + np.array(nn)
            node.post_rate = node.prior_rate.expected_x()[uu] + np.array(bl * F * G)
        else:
            node.post_shape = node.prior_shape + np.array(nn)
            node.post_rate = node.prior_rate + np.array(bl * F * G)
            # print "Min, Mean, Max of post shape: {}\t{}\t{}******".format(np.min(node.post_shape),
            #                                                               np.mean(node.post_shape),
            #                                                               np.max(node.post_shape))
            # print "Min, Mean, Max of post rate:  {}\t{}\t{}******".format(np.min(node.post_rate),
            #                                                               np.mean(node.post_rate),
            #                                                               np.max(node.post_rate))

    def update_overdispersion_natural(self):
        node = self.nodes['overdispersion_natural']
        uu = self.Nframe['unit']
        bl = self.nodes['baseline'].expected_x()[uu]
        F = self.F_prod()
        G = self.G_prod()
        time_nat = self.time_natural

        # update of post_shape
        cnt_dframe = self.Nframe.sort_values(['unit', 'trial', 'time'])['count']
        # index matrix of 2 columns: Column 1 original, Column 2 sorted
        ind_mat = np.c_[np.array(cnt_dframe.index), np.array(xrange(cnt_dframe.index.shape[0]))]

        # cumulative sum from data
        cnt_cumsum = np.cumsum(np.array(cnt_dframe).reshape(-1, time_nat)[:, ::-1], axis=1)[:, ::-1]
        expected_phi = node.expected_x()
        # sort expected values of phi
        exphi_sorted = self._sort_values(ind_mat, expected_phi)
        # sort baseline
        bl_sorted = self._sort_values(ind_mat, bl)

        # sort F values
        if not hasattr(F, '__iter__'):
            F_sorted = F * np.ones(self.M)
        else:
            F_sorted = self._sort_values(ind_mat, F)

        # sort G values
        if not hasattr(G, '__iter__'):
            G_sorted = G * np.ones(self.M)
        else:
            G_sorted = self._sort_values(ind_mat, G)

        cumprod_phi = np.cumprod(exphi_sorted.reshape(-1, time_nat), axis=1) / exphi_sorted.reshape(-1, time_nat)
        prod_phi_F = cumprod_phi * bl_sorted.reshape(-1, time_nat) * F_sorted.reshape(-1, time_nat) * \
                     G_sorted.reshape(-1, time_nat)
        cumsum_phi_F = np.cumsum(prod_phi_F[:, ::-1], axis=1)[:, ::-1]
        # cumprod_phi = np.cumsum(np.log(exphi_sorted).reshape(-1, time_nat), axis=1)
        # prod_phi_F = cumprod_phi - np.log(exphi_sorted).reshape(-1, time_nat) + \
        #              np.log(bl_sorted).reshape(-1, time_nat) + \
        #              np.log(F_sorted).reshape(-1, time_nat) + \
        #              np.log(G_sorted).reshape(-1, time_nat)
        # cumsum_phi_F = np.cumsum(np.exp(prod_phi_F)[:, ::-1], axis=1)[:, ::-1]

        # create an array for update rule
        reg_prod = np.ones(node.prior_rate.reshape(-1, time_nat).shape[0])

        # create sorted copies of post_rate and post_shape
        postrate_sorted = self._sort_values(ind_mat, node.post_rate).reshape(-1, time_nat)
        postshape_sorted = self._sort_values(ind_mat, node.post_shape).reshape(-1, time_nat)

        # create sorted copies of prior_rate and prior_shape
        priorate_sorted = self._sort_values(ind_mat, node.prior_rate).reshape(-1, time_nat)
        priorshape_sorted = self._sort_values(ind_mat, node.prior_shape).reshape(-1, time_nat)

        print "zeta priors: {}, {}, {}".format(node.prior_rate.min(),
                                               node.prior_rate.mean(),
                                               node.prior_rate.max())
        print "omega priors: {}, {}, {}".format(node.prior_shape.min(),
                                                node.prior_shape.mean(),
                                                node.prior_shape.max())

        print "Old zeta: {}, {}, {}".format(postrate_sorted.min(),
                                            postrate_sorted.mean(),
                                            postrate_sorted.max())
        print "Old omega: {}, {}, {}".format(postshape_sorted.min(),
                                             postshape_sorted.mean(),
                                             postshape_sorted.max())

        print "Min, Mean, Max of cumsum_phi_F:  {}\t{}\t{} @@@@@@".format(np.min(cumsum_phi_F),
                                                                          np.mean(cumsum_phi_F),
                                                                          np.max(cumsum_phi_F))

        if node.has_parents:
            print "Overdispersion_natural has parents!"
            prior_rate = node.prior_rate.expected_x()[uu]
            prior_shape = node.prior_shape.expected_x()[uu]

            # update post shape and post rate
            node.post_shape, node.post_rate = _reg_omega_zeta(time_nat, 
                node.post_shape.reshape(-1, time_nat), node.post_rate.reshape(-1, time_nat), 
                new_omega, new_zeta, prior_shape, prior_rate, cnt_cumsum, cumsum_phi_F, reg_prod)

            # for i in range(time_nat):
            #     new_zeta[:, i] = cumsum_phi_F[:, i] * reg_prod + prior_rate.reshape(-1, time_nat)[:, i]
            #     new_omega[:, i] = cnt_cumsum[:, i] + prior_shape.reshape(-1, time_nat)[:, i]

            #     reg_prod *= (node.post_rate.reshape(-1, time_nat)[:, i] /
            #                  node.post_shape.reshape(-1, time_nat)[:, i]) * (new_omega[:, i] / new_zeta[:, i])

        else:
            for i in range(time_nat):
                # create a temp vector to keep last ratio of rate / shape
                temp_ratio = postrate_sorted[:, i] / postshape_sorted[:, i]
                # update post rate and post shape
                postrate_sorted[:, i] = cumsum_phi_F[:, i] * reg_prod + priorate_sorted[:, i]
                postshape_sorted[:, i] = cnt_cumsum[:, i] + priorshape_sorted[:, i]
                # update regressive product
                reg_prod *= temp_ratio * (postshape_sorted[:, i] / postrate_sorted[:, i])

        node.post_shape = self._unsort_values(ind_mat, postshape_sorted.ravel())
        node.post_rate = self._unsort_values(ind_mat, postrate_sorted.ravel())

            # # update post shape and post rate
            # node.post_shape, node.post_rate = _reg_omega_zeta(time_nat,
            #     node.post_shape.reshape(-1, time_nat), node.post_rate.reshape(-1, time_nat),
            #     new_omega, new_zeta,
            #     node.prior_shape.reshape(-1, time_nat), node.prior_rate.reshape(-1, time_nat),
            #     reg_prod, cnt_cumsum, cumsum_phi_F)

        print "Min, Mean, Max of post shape: {}\t{}\t{}******".format(np.min(node.post_shape),
                                                                      np.mean(node.post_shape),
                                                                      np.max(node.post_shape))
        print "Min, Mean, Max of post rate:  {}\t{}\t{}******".format(np.min(node.post_rate),
                                                                      np.mean(node.post_rate),
                                                                      np.max(node.post_rate))


        # # update of post_shape
        # cnt_array = self.Nframe.sort_values(['unit', 'trial', 'time'])['count'].values
        # cnt_cumsum = np.cumsum(cnt_array.reshape(-1, time_nat)[:, ::-1], axis=1)[:, ::-1]

        # ####### Correct! Cross-validated by simple loops ###################
        # expected_phi = node.expected_x()

        # cumprod_phi = np.cumsum(np.log(expected_phi).reshape(-1, time_nat), axis=1)
        # prod_phi_F = cumprod_phi - np.log(expected_phi).reshape(-1, time_nat) + \
        #              (np.log(bl).reshape(-1, time_nat) + np.log(F).reshape(-1, time_nat) + np.log(G))

        # cumsum_phi_F = np.cumsum(np.exp(prod_phi_F)[:, ::-1], axis=1)[:, ::-1]

        # #reg_prod = np.ones(cumsum_phi_F.reshape(-1, time_nat).shape[0])
        # reg_prod = np.ones(node.prior_rate.reshape(-1, time_nat).shape[0])
        # new_zeta = node.post_rate.reshape(-1, time_nat).copy()
        # new_omega = node.post_shape.reshape(-1, time_nat).copy()
            
        # # print "Old zeta: {}, {}, {}".format(new_zeta.min(),
        # #                                     new_zeta.mean(),
        # #                                     new_zeta.max())
        # # print "Old omega: {}, {}, {}".format(new_omega.min(),
        # #                                      new_omega.mean(),
        # #                                      new_omega.max())

        # if node.has_parents:
        #     print "Overdispersion_natural has parents!"
        #     prior_rate = node.prior_rate.expected_x()[uu]
        #     prior_shape = node.prior_shape.expected_x()[uu]

        #     # update post shape and post rate
        #     node.post_shape, node.post_rate = _reg_omega_zeta(time_nat, 
        #         node.post_shape.reshape(-1, time_nat), node.post_rate.reshape(-1, time_nat), 
        #         new_omega, new_zeta, prior_shape, prior_rate, cnt_cumsum, cumsum_phi_F, reg_prod)

        #     # for i in range(time_nat):
        #     #     new_zeta[:, i] = cumsum_phi_F[:, i] * reg_prod + prior_rate.reshape(-1, time_nat)[:, i]
        #     #     new_omega[:, i] = cnt_cumsum[:, i] + prior_shape.reshape(-1, time_nat)[:, i]

        #     #     reg_prod *= (node.post_rate.reshape(-1, time_nat)[:, i] /
        #     #                  node.post_shape.reshape(-1, time_nat)[:, i]) * (new_omega[:, i] / new_zeta[:, i])

        # else:
        # #     for i in range(time_nat):
        # #         new_zeta[:, i] = cumsum_phi_F[:, i] * reg_prod + node.prior_rate.reshape(-1, time_nat)[:, i]
        # #         new_omega[:, i] = cnt_cumsum[:, i] + node.prior_shape.reshape(-1, time_nat)[:, i]

        # #         reg_prod *= (node.post_rate.reshape(-1, time_nat)[:, i] /
        # #                      node.post_shape.reshape(-1, time_nat)[:, i]) * (new_omega[:, i] / new_zeta[:, i])

        # # node.post_shape = new_omega.ravel()
        # # node.post_rate = new_zeta.ravel()

        #     # update post shape and post rate
        #     node.post_shape, node.post_rate = _reg_omega_zeta(time_nat, 
        #         node.post_shape.reshape(-1, time_nat), node.post_rate.reshape(-1, time_nat), 
        #         new_omega, new_zeta, 
        #         node.prior_shape.reshape(-1, time_nat), node.prior_rate.reshape(-1, time_nat),
        #         reg_prod, cnt_cumsum, cumsum_phi_F)

        # # print "Min, Mean, Max of post shape: {}\t{}\t{}******".format(np.min(node.post_shape),
        # #                                                               np.mean(node.post_shape),
        # #                                                               np.max(node.post_shape))
        # # print "Min, Mean, Max of post rate:  {}\t{}\t{}******".format(np.min(node.post_rate),
        # #                                                               np.mean(node.post_rate),
        # #                                                               np.max(node.post_rate))


    def finalize(self):
        """
        This should be called once all the relevant variables are initialized.
        """
        if 'baseline' in self.nodes:
            self.baseline = True
            self.nodes['baseline'].update = self.update_baseline

        if {'HMM', 'fr_latents'}.issubset(self.nodes):
            self.latents = True
            self.F_prod(update=True)
            self.nodes['fr_latents'].update = self.update_fr_latents

            self.nodes['HMM'].update_finalizer = (
                lambda idx: self.F_prod(idx, update=True))
        else:
            self.latents = False

        if 'fr_regressors' in self.nodes:
            self.regressors = True
            self.G_prod(update=True)
            self.nodes['fr_regressors'].update = self.update_fr_regressors
        else:
            self.regressors = False

        if 'overdispersion' in self.nodes:
            self.overdispersion = True
            self.nodes['overdispersion'].update = self.update_overdispersion
        else:
            self.overdispersion = False

        if 'overdispersion_natural' in self.nodes:
            self.overdispersion_natural = True
            self.nodes['overdispersion_natural'].update = self.update_overdispersion_natural
        else:
            self.overdispersion_natural = False

        return self


    @jit
    def F_prod(self, k=None, update=False):
        """
        Accessor method to return the value of the F product.
        If k is specified, return F_{mk} (product over all but k),
        else return F_{m} (product over all k). If update=True,
        recalculate F before returning the result and cache the new
        matrix. Returned array is one row per observation.
        """
        if not self.latents:
            return 1

        if update:
            uu = self.Nframe['unit']
            tt = self.Nframe['time']

            xi = self.nodes['HMM'].nodes['z'].z[1]
            lam = self.nodes['fr_latents'].expected_x()
            if k is not None:
                zz = xi[tt, k]
                ww = lam[uu, k]
                # vv = ne.evaluate("1 - zz + zz * w")
                vv = 1 - zz + zz * ww
                self._Fpre[:, k] = vv
            else:
                zz = xi[tt]
                ww = lam[uu]
                # vv = ne.evaluate("1 - zz + zz * w")
                vv = 1 - zz + zz * ww
                self._Fpre = vv

            # work in log space to avoid over/underflow
            Fpre = self._Fpre
            dd = np.sum(np.log(Fpre), axis=1)
            self._F = np.exp(dd)
            ddd = dd[:, np.newaxis]
            self._Fk = np.exp(ddd - np.log(Fpre))

        if k is not None:
            return self._Fk[..., k]
        else:
            return self._F

    @jit
    def G_prod(self, k=None, update=False):
        """
        Return the value of the G product.
        If k is specified, return G_{mk} (product over all but k),
        else return G_{m} (product over all k). If update=True,
        recalculate G before returning the result and cache the new
        matrix.

        NOTE: Because the regressors may vary by *presentation* and not
        simply by movie time, the regressors are in a "melted" dataframe
        with each (unit, presentation) pair in a row by itself. As a
        result, X is (M, J), G_{tu} is (M,), and G_{tku} is (M, J).
        """
        if not self.regressors:
            return 1

        if update:
            uu = self.Nframe['unit']

            lam = self.nodes['fr_regressors'].expected_x()
            if k is not None:
                zz = lam[uu][k]
                # get x values for kth regressor
                xx = self.Xframe.values[:, k]
                # vv = ne.evaluate("zz ** xx")
                vv = zz ** xx
                self._Gpre[:, k] = vv
            else:
                zz = lam[uu]
                # get x values for kth regressor; col 0 = time
                xx = self.Xframe.values
                # vv = ne.evaluate("zz ** xx")
                vv = zz ** xx
                self._Gpre = vv

            # work in log space to avoid over/underflow
            Gpre = self._Gpre
            # dd = ne.evaluate("sum(log(Gpre), axis=1)")
            dd = np.sum(np.log(Gpre), axis=1)
            # self._G = ne.evaluate("exp(dd)")
            self._G = np.exp(dd)
            ddd = dd[:, np.newaxis]
            # self._Gk = ne.evaluate("exp(ddd - log(Gpre))")
            self._Gk = np.exp(ddd - np.log(Gpre))

        if k is not None:
            return self._Gk[..., k]
        else:
            return self._G

    def L(self, keeplog=False, print_pieces=False):
        """
        Calculate E[log p - log q] in the variational approximation.
        This result is only valid immediately after the E-step (i.e, updating
        E[z] in forward-backward). For a discussion of cancellations that
        occur in this context, cf. Beal (2003) ~ (3.79).
        """
        Elogp = self.expected_log_evidence()  # observation model
        H = 0

        if print_pieces:
            print
            print "xxxxxxxxxxxxxxxxxxxxxxxxxx"
            print "Elogp = {}\tH = {}".format(Elogp, H)

        for _, node in self.nodes.iteritems():
            logp = node.expected_log_prior()
            logq = node.entropy()
            if print_pieces:
                print "{}: \n \tElogp = {}\tH = {}\tsum = {}".format(
                    node.name, logp, logq, logp + logq)

            Elogp += logp
            H += logq

        L = Elogp + H

        if keeplog:
            self.log['L'].append(L)
        if print_pieces:
            print "xxxxxxxxxxxxxxxxxxxxxxxxxx"
            print

        return L

    def iterate(self, verbosity=0, keeplog=False):
        """
        Do one iteration of variational inference, updating each chain in turn.
        verbosity is a verbosity level:
            0: no print to screen
            1: print L value on each iteration
            2: print L value each update during each iteration
            3: print all pieces of L
        keeplog = True does internal logging for debugging; values are kept in
            the dict self.log
        """
        doprint = verbosity > 1
        print_pieces = verbosity > 2
        calc_L = doprint or keeplog

        if len(self.Lvalues) > 0:
            lastL = self.Lvalues[-1]
        else:
            lastL = -np.inf

        # M step
        if self.baseline:
            self.nodes['baseline'].update()
            if self.nodes['baseline'].has_parents:
                self.nodes['baseline'].update_parents()
            if calc_L:
                Lval = self.L(keeplog=keeplog, print_pieces=print_pieces)
                assert((Lval >= lastL) | np.isclose(Lval, lastL))
                lastL = Lval
            if doprint:
                print "         updated baselines: L = {}".format(Lval)

        if self.latents:
            # for k in np.random.permutation(self.K):
            for k in xrange(self.K):
            # for k in range(1):
                # M step
                self.nodes['fr_latents'].update(k)
                if self.nodes['fr_latents'].has_parents:
                    self.nodes['fr_latents'].update_parents(k)
                if calc_L:
                    Lval = self.L(keeplog=keeplog, print_pieces=print_pieces)
                    assert((Lval >= lastL) | np.isclose(Lval, lastL))
                    lastL = Lval
                if doprint:
                    print("chain {}: updated firing rate effects: L = {}"
                          ).format(k, Lval)

                # E step
                logpsi = self.calc_log_evidence(k)
                self.nodes['HMM'].update(k, logpsi)
                if calc_L:
                    Lval = self.L(keeplog=keeplog, print_pieces=print_pieces)
                    assert((Lval >= lastL) | np.isclose(Lval, lastL))
                    lastL = Lval
                if doprint:
                    print "chain {}: updated z: L = {}".format(k, Lval)

        if self.regressors:
            self.nodes['fr_regressors'].update()
            if self.nodes['fr_regressors'].has_parents:
                self.nodes['fr_regressors'].update_parents()
            if calc_L:
                Lval = self.L(keeplog=keeplog, print_pieces=print_pieces)
                assert((Lval >= lastL) | np.isclose(Lval, lastL))
                lastL = Lval
            if doprint:
                print "         updated regressor effects: L = {}".format(Lval)

        if self.overdispersion:
            self.nodes['overdispersion'].update()
            if self.nodes['overdispersion'].has_parents:
                self.nodes['overdispersion'].update_parents()
            if calc_L:
                Lval = self.L(keeplog=keeplog, print_pieces=print_pieces)
                assert((Lval >= lastL) | np.isclose(Lval, lastL))
                lastL = Lval
            if doprint:
                print("         updated overdispersion effects: L = {}"
                      ).format(Lval)

        if self.overdispersion_natural:
            self.nodes['overdispersion_natural'].update()
            if self.nodes['overdispersion_natural'].has_parents:
                self.nodes['overdispersion_natural'].update_parents()
            if calc_L:
                Lval = self.L(keeplog=keeplog, print_pieces=print_pieces)
                if not (Lval >= lastL) | np.isclose(Lval, lastL):
                    print "!!!!!!ERROR ERROR ERROR ERROR ERROR!!!!!!"
                # assert((Lval >= lastL) | np.isclose(Lval, lastL))
                lastL = Lval
            if doprint:
                print("         updated overdispersion_natural effects: L = {}"
                      ).format(Lval)


    def do_inference(self, verbosity=0, tol=1e-3, keeplog=False,
                     maxiter=np.inf, delayed_iters=[]):
        """
        Perform variational inference by minimizing free energy.
        delayed_iters is a dict of variable names that should not be
        iterated over in the early going; once the algorithm has converged,
        these variables are added in and inference run a second time
        """
        self.Lvalues.append(self.L())
        delta = 1
        idx = 0

        while np.abs(delta) > tol and idx < maxiter:
            if verbosity > 0:
                print "Iteration {}: L = {}".format(idx, self.Lvalues[-1])
                print "delta = " + str(delta)

            self.iterate(verbosity=verbosity, keeplog=keeplog)
            self.Lvalues.append(self.L())

            delta = ((self.Lvalues[-1] - self.Lvalues[-2]) /
                     np.abs(self.Lvalues[-1]))
            if not (delta > 0) | np.isclose(delta, 0):
                print "!!!!!!ERROR ERROR ERROR ERROR ERROR!!!!!!"
                print "Lval previous, Lval current: {}, {}".format(
                    self.Lvalues[-2], self.Lvalues[-1])
                print "delta value: {}".format(delta)
                return
            #assert((delta > 0) | np.isclose(delta, 0))
            idx += 1

        # now redo inference, this time including all variables that
        # were delayed
        if len(delayed_iters) > 0:
            print "Initial optimization done, adding {}".format(
                ', '.join(delayed_iters))
            self.do_inference(verbosity=verbosity, tol=tol, keeplog=keeplog,
                              maxiter=maxiter, delayed_iters=[])


@jit
def exact_minfun(epsilon, aa, ww, uu, Fblod, X):
    U, R = aa.shape
    M = X.shape[0]
    eps = epsilon.reshape(U, R)
    grad = np.empty((U, R))

    G = np.zeros(M)

    elbo = _minfun_guts(eps, grad, G, aa, ww, uu, Fblod, X)
    assert (np.isfinite(np.log(-elbo)))

    return np.log(-elbo), grad.ravel() / elbo


@jit(nopython=True, nogil=True)
def _minfun_guts(eps, grad, G, aa, ww, uu, Fblod, X):
    U, R = aa.shape
    M = X.shape[0]
    Uw = ww.shape[0]
    elbo = 0.0

    # calculate G
    for m in xrange(M):
        log_G = 0.0
        for r in xrange(R):
            log_G += eps[uu[m], r] * X[m, r]
        G[m] = np.exp(log_G)

    # calculate (U, R) piece of elbo and grad
    for u in xrange(U):
        for r in xrange(R):
            if Uw == 1:
                w_exp_eps = ww[0, r] * np.exp(eps[u, r])
            else:
                w_exp_eps = ww[u, r] * np.exp(eps[u, r])

            elbo += aa[u, r] * eps[u, r]
            elbo -= w_exp_eps
            grad[u, r] = aa[u, r] - w_exp_eps

    # calculate flat piece of elbo and grad
    for m in xrange(M):
        FblodG = Fblod[m] * G[m]
        elbo -= FblodG
        for r in xrange(R):
            grad[uu[m], r] -= FblodG * X[m, r]

    return elbo


@jit(nopython=True, nogil=True)
def _reg_omega_zeta(time_nat, omega, zeta, new_omega, new_zeta, prior_shape, prior_rate, 
    reg_prod, cnt_cumsum, cumsum_phi_F):
    
    for i in xrange(time_nat):
        new_zeta[:, i] = cumsum_phi_F[:, i] * reg_prod + prior_rate[:, i]
        new_omega[:, i] = cnt_cumsum[:, i] + prior_shape[:, i]

        for j in xrange(reg_prod.shape[0]):
            reg_prod[j] *= (zeta[j, i] / omega[j, i]) * (new_omega[j, i] / new_zeta[j, i])
    return new_omega.ravel(), new_zeta.ravel()
