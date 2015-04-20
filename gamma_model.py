from __future__ import division
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numexpr as ne
import spiketopics.nodes as nd

class GammaModel:
    """
    This class fits a Poisson observation model using a product of Gamma-
    distributed variables to model the rates.

    N ~ Poiss(\mu)
    \mu = prod \lambda
    \lambda ~ Gamma
    """

    def __init__(self, data, K):
        """
        Construct model object.

        data: Pandas DataFrame with one row per observation and columns
            'unit', 'time', and 'count' (time is stimulus time)
        K: number of latent categories to infer
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

        self.nodes = {}  # dict for variable nodes in graphical model

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

        cols = ['unit', 'time', 'count']
        self.Nframe = data[cols].copy()
        self.Xframe = data.drop(cols, axis=1).copy()

        # make sure units are indexed from 0
        self.Nframe['unit'] = self.Nframe['unit'] - np.min(self.Nframe['unit'])

        # make arrays:
        # array of counts for each time, unit
        countframe = self.Nframe.groupby(['time', 'unit']).sum().unstack(level=1)
        countarr = countframe.values
        self.N = np.ma.masked_where(np.isnan(countarr), countarr).astype('int')

        # array of observations at each time, unit
        Nobs =  self.Nframe.groupby(['time', 'unit']).count().unstack()
        self.Nobs = np.ma.masked_where(np.isnan(Nobs), Nobs).astype('int')

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
        if 'prior_shape_shape' in kwargs:
            nodes = nd.initialize_gamma_hierarchy(name, parent_shape,
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
            (), **kwargs)

        return self

    def initialize_latents(self, **kwargs):
        nodes = nd.initialize_HMM(self.K, 2, self.T, **kwargs)

        self.nodes.update({n.name: n for n in nodes})

        return self

    def update_baseline(self):
        node = self.nodes['baseline']

        if self.overdispersion:
            uu = self.Nframe['unit']
            tt = self.Nframe['time']
            od = self.nodes['overdispersion'].expected_x()
            F = self.F_prod(flat=True)
            G = self.G_prod(flat=True)
            allprod = od * F * G
            eff_rate = pd.DataFrame(allprod).groupby(uu).sum().values.squeeze()
        else:
            F = self.F_prod() 
            G = self.G_prod()
            allprod = F * G
            eff_rate = np.sum(allprod, axis=0)

        node.post_shape = (node.prior_shape.expected_x() + 
            np.sum(self.N, axis=0).data)
        node.post_rate = node.prior_rate.expected_x() + eff_rate

    def update_fr_latents(self, idx):
        lam = self.nodes['fr_latents']
        xi = self.nodes['HMM'].nodes['z'].z[1]
        Nz = xi[:, idx].dot(self.N)

        if self.overdispersion:
            uu = self.Nframe['unit']
            tt = self.Nframe['time']
            od = self.nodes['overdispersion'].expected_x()
            bl = self.nodes['baseline'].expected_x()[uu]
            Fz = self.F_prod(idx, flat=True) * xi[tt, idx]
            G = self.G_prod(flat=True)
            allprod = bl * od * Fz * G
            eff_rate = pd.DataFrame(allprod).groupby(uu).sum().values.squeeze()
        else:
            bl = self.nodes['baseline'].expected_x()
            Fz = self.F_prod(idx) * xi[:, np.newaxis, idx] 
            G = self.G_prod()
            allprod = bl * Fz * G
            eff_rate = np.sum(allprod, axis=0)

        lam.post_shape[..., idx] = lam.prior_shape.expected_x()[..., idx] + Nz
        lam.post_rate[..., idx] = lam.prior_rate.expected_x()[..., idx] + eff_rate
        self.F_prod(update=True)

    def calc_log_evidence(self, idx):
        """
        Calculate p(N|z, theta) for use in updating HMM. Need only be
        correct up to an overall constant.
        """ 
        logpsi = np.empty((self.T, 2))

        lam = self.nodes['fr_latents']
        if self.overdispersion:
            N = self.Nframe
            nn = N['count']
            uu = N['unit']
            od = self.nodes['overdispersion'].expected_x()
            bl = self.nodes['baseline'].expected_x()[uu]
            Fk = self.F_prod(idx, flat=True)
            G = self.G_prod(flat=True)
            Gbar = np.mean(G)  # helps if we use this as a normalizer
            allprod = -bl * od * Fk * G / Gbar
            bar_log_lam = lam.expected_log_x()[idx, uu]

            N['lam0'] = allprod
            N['lam1'] = allprod + (nn *  bar_log_lam/ Gbar)

            logpsi = N.groupby('time').sum()[['lam0', 'lam1']].values

        else:
            bl = self.nodes['baseline'].expected_x()
            Fk = self.F_prod(idx)
            G = self.G_prod()
            Gbar = np.mean(G)
            allprod = -np.sum(bl * Fk * G / Gbar, axis=1)
            bar_log_lam = lam.expected_log_x()[..., idx]

            logpsi[:, 0] = allprod
            logpsi[:, 1] = allprod + np.sum(self.N * bar_log_lam, axis=1)

        return logpsi

    def expected_log_evidence(self):
        """
        Calculate E[log p(N, z|rest).
        """
        uu = self.Nframe['unit']
        nn = self.Nframe['count']

        Elogp = 0
        eff_rate = 1

        if self.baseline:
            node = self.nodes['baseline']
            bar_log_lam = node.expected_log_x()
            bar_lam = node.expected_x()

            Elogp += np.sum(self.N * bar_log_lam[np.newaxis, :])
            eff_rate *= bar_lam[uu]

        if self.latents:
            node = self.nodes['fr_latents']
            bar_log_lam = node.expected_log_x()
            xi = self.nodes['HMM'].nodes['z'].z[1]

            Elogp += np.sum(self.N[..., np.newaxis] * xi[..., np.newaxis, :] *
                bar_log_lam[np.newaxis, ...])
            eff_rate *= self.F_prod(flat=True)

            # pieces for A and pi
            Elogp += self.nodes['HMM'].expected_log_state_sequence()

        if self.regressors:
            node = self.nodes['fr_regressors']
            bar_log_lam = node.expected_log_x()
            xx = self.Xframe.values

            Elogp += np.sum(nn[:, np.newaxis] * xx * bar_log_lam[uu])
            eff_rate *= self.G_prod(flat=True)

        if self.overdispersion:
            node = self.nodes['overdispersion']
            bar_log_lam = node.expected_log_x()
            bar_lam = node.expected_x()

            Elogp += nn.dot(bar_log_lam)
            eff_rate *= bar_lam

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
        aa = self.nodes['fr_regressors'].post_shape
        minfun = self._make_exact_minfun()

        eps_starts = np.log(aa / starts)
        res = minimize(minfun, eps_starts, jac=True)
        if not res.success:
            print "Warning: optimization terminated without success."
            print res.message
        eps = res.x.reshape(self.U, self.R)
        bb = aa * np.exp(-eps)
        return bb

    def _make_exact_minfun(self):
        """
        Factory function that returns a function to be minimized.
        This version uses an exact minimization objective.
        """
        uu = self.Nframe['unit']
        if self.overdispersion:
            od = self.nodes['overdispersion'].expected_x()
        else:
            od = 1
        F = self.F_prod(flat=True)
        bl = self.nodes['baseline'].expected_x()[uu]

        aa = self.nodes['fr_regressors'].post_shape
        ww = self.nodes['fr_regressors'].prior_rate.expected_x().reshape(-1, self.R)

        def minfun(epsilon): 
            """
            This is the portion of the evidence lower bound that depends on 
            the b parameter. eps = log(a/b)
            """
            eps = epsilon.reshape(self.U, self.R)
            sum_log_G = np.sum(eps[uu] * self.Xframe.values, axis=1)
            G = np.exp(sum_log_G)

            elbo = np.sum(aa * eps)
            elbo += -np.sum(ww * np.exp(eps))
            FthG = (bl * od * F * G).view(np.ndarray)
            elbo += -np.sum(FthG)

            # grad = grad(elbo)
            grad = aa - ww * np.exp(eps)
            grad -= (FthG[:, np.newaxis] * self.Xframe).groupby(uu).sum().values

            # minimization objective is log(-elbo)
            return np.log(-elbo), grad.ravel() / elbo

        return minfun

    def update_overdispersion(self):
        node = self.nodes['overdispersion']
        nn = self.Nframe['count']
        uu = self.Nframe['unit']
        bl = self.nodes['baseline'].expected_x()[uu]
        F = self.F_prod(flat=True)
        G = self.G_prod(flat=True)

        node.post_shape = node.prior_shape + nn
        node.post_rate = node.prior_rate + bl * F * G

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

        return self

    def F_prod(self, k=None, update=False, flat=False):
        """
        Accessor method to return the value of the F product.
        If k is specified, return F_{tku} (product over all but k), 
        else return F_{tu} (product over all k). If update=True, 
        recalculate F before returning the result and cache the new 
        matrix. If flat=True, return the one-row-per-observation 
        version of F_{tu}.
        """
        if update:
            uu = self.Nframe['unit']
            tt = self.Nframe['time']

            xi = self.nodes['HMM'].nodes['z'].z[1]
            lam = self.nodes['fr_latents'].expected_x()
            if k is not None:
                zz = xi[tt, k]
                w = lam[uu, k]
                vv = ne.evaluate("1 - zz + zz * w")
                self._Fpre[:, k] = vv
            else:
                zz = xi[tt]
                w = lam[uu]
                vv = ne.evaluate("1 - zz + zz * w")
                self._Fpre = vv

            # work in log space to avoid over/underflow
            Fpre = self._Fpre
            dd = ne.evaluate("sum(log(Fpre), axis=1)")
            self._Ftu_flat = ne.evaluate("exp(dd)")
            ddd = dd[:, np.newaxis]
            self._Ftuk_flat = ne.evaluate("exp(ddd - log(Fpre))")

            # make non-flat versions
            Ftu = pd.DataFrame(self._Ftu_flat).groupby([tt, uu]).sum()
            self._Ftu = Ftu.values.reshape(self.T, self.U)
            Ftuk = pd.DataFrame(self._Ftuk_flat).groupby([tt, uu]).sum()
            self._Ftuk = Ftuk.values.reshape(self.T, self.U, -1)

        if k is not None:
            if flat:
                return self._Ftuk_flat[..., k]
            else:
                return self._Ftuk[..., k]
        else:
            if flat:
                return self._Ftu_flat
            else:
                return self._Ftu

    def G_prod(self, k=None, update=False, flat=False):
        """
        Return the value of the G product.
        If k is specified, return G_{tku} (product over all but k),
        else return G_{tu} (product over all k). If update=True,
        recalculate G before returning the result and cache the new 
        matrix.
        
        NOTE: Because the regressors may vary by *presentation* and not 
        simply by movie time, the regressors are in a "melted" dataframe
        with each (unit, presentation) pair in a row by itself. As a 
        result, X is (M, J), G_{tu} is (M,), and G_{tku} is (M, J).
        """
        if update:
            uu = self.Nframe['unit']
            tt = self.Nframe['time']

            lam = self.nodes['fr_regressors'].expected_x()
            if k is not None:
                zz = lam[uu][k]
                # get x values for kth regressor
                xx = self.Xframe.values[:, k]
                vv = ne.evaluate("zz ** xx")
                self._Gpre[:, k] = vv
            else:
                zz = lam[uu]
                # get x values for kth regressor; col 0 = time
                xx = self.Xframe.values
                vv = ne.evaluate("zz ** xx")
                self._Gpre = vv

            # work in log space to avoid over/underflow
            Gpre = self._Gpre
            dd = ne.evaluate("sum(log(Gpre), axis=1)")
            self._Gtu_flat = ne.evaluate("exp(dd)")
            ddd = dd[:, np.newaxis]
            self._Gtuk_flat = ne.evaluate("exp(ddd - log(Gpre))")

            # make non-flat versions
            Gtu = pd.DataFrame(self._Gtu_flat).groupby([tt, uu]).sum()
            self._Gtu = Gtu.values.reshape(self.T, self.U)
            Gtuk = pd.DataFrame(self._Gtuk_flat).groupby([tt, uu]).sum()
            self._Gtuk = Gtuk.values.reshape(self.T, self.U, -1)

        if k is not None:
            if flat:
                return self._Gtuk_flat[..., k]
            else:
                return self._Gtuk[..., k]
        else:
            if flat:
                return self._Gtu_flat
            else:
                return self._Gtu

    def L(self, keeplog=False):
        """
        Calculate E[log p - log q] in the variational approximation.
        This result is only valid immediately after the E-step (i.e, updating
        E[z] in forward-backward). For a discussion of cancellations that 
        occur in this context, cf. Beal (2003) ~ (3.79).
        """
        Elogp = self.expected_log_evidence()  # observation model
        H = 0

        for _, node in self.nodes.iteritems():
            Elogp += node.expected_log_prior()
            H += node.entropy()

        L = Elogp + H

        if keeplog:
            self.log['L'].append(L)

        return L

    def iterate(self, verbosity=0, keeplog=False):
        """
        Do one iteration of variational inference, updating each chain in turn.
        verbosity is a verbosity level:
            0: no print to screen
            1: print L value on each iteration
            2: print L value each update during each iteration
        keeplog = True does internal logging for debugging; values are kept in
            the dict self.log
        """
        doprint = verbosity > 1 
        calc_L = doprint or keeplog
        
        # M step
        if self.baseline:
            self.nodes['baseline'].update()
            if self.nodes['baseline'].has_parents:
                self.nodes['baseline'].update_parents()
            if calc_L:
                Lval = self.L(keeplog=keeplog) 
            if doprint:
                print "         updated baselines: L = {}".format(Lval)

        if self.latents:
            for k in xrange(self.K):
                self.nodes['fr_latents'].update(k)
                if self.nodes['fr_latents'].has_parents:
                    self.nodes['fr_latents'].update_parents(k)
                if calc_L:
                    Lval = self.L(keeplog=keeplog) 
                if doprint:
                    print ("chain {}: updated firing rate effects: L = {}"
                        ).format(k, Lval)

        if self.regressors:
            self.nodes['fr_regressors'].update()
            if self.nodes['fr_regressors'].has_parents:
                self.nodes['fr_regressors'].update_parents()
            if calc_L:
                Lval = self.L(keeplog=keeplog) 
            if doprint:
                print "         updated regressor effects: L = {}".format(Lval)

        if self.overdispersion:
            self.nodes['overdispersion'].update()
            if self.nodes['overdispersion'].has_parents:
                self.nodes['overdispersion'].update_parents()
            if calc_L:
                Lval = self.L(keeplog=keeplog) 
            if doprint:
                print ("         updated overdispersion effects: L = {}"
                    ).format(Lval)

        # E step        
        if self.latents:
            for k in xrange(self.K):
                logpsi = self.calc_log_evidence(k)
                self.nodes['HMM'].update(k, logpsi)
                if calc_L:
                    Lval = self.L(keeplog=keeplog) 
                if doprint:
                    print "chain {}: updated z: L = {}".format(k, Lval)

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
            assert((delta > 0) | np.isclose(delta, 0))
            idx += 1 

        # now redo inference, this time including all variables that 
        # were delayed
        if len(delayed_iters) > 0:
            print "Initial optimization done, adding {}".format(', '.join(delayed_iters))
            self.do_inference(verbosity=verbosity, tol=tol, keeplog=keeplog,
                maxiter=maxiter, delayed_iters=[])
