from __future__ import division
import numpy as np
import pandas as pd
from scipy.optimize import minimize, line_search
import numexpr as ne
import spiketopics.nodes as nd

class LogNormalModel:
    """
    This class fits a Poisson observation model using a gaussian distribution
    for log firing rate.

    N ~ Poiss(exp(eta))
    eta = lambda_0 + z * b + x * beta
    lambda_0, b, beta ~ Normal
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
        self.maxiter = 3  # number of BFGS iterations per variable update

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

    def _initialize_gaussian_nodes(self, name, node_shape, parent_shape,  
        grandparent_shape, **kwargs):
        """
        Perform multiple distpatch for initializing gaussian nodes based on 
            parameters in kwargs.
        name: name of the base node
        node_shape: shape of base node
        parent_shape: shape of parent node
        grandparent_shape: dimension of parent prior params
        """
        # test if parameters indicate there's a hierarchy
        if 'prior_prec_shape' in kwargs:
            nodes = nd.initialize_gaussian_hierarchy(name, node_shape,
                parent_shape, grandparent_shape, **kwargs)
        else:
            nodes = nd.initialize_gaussian(name, node_shape, 
                parent_shape, **kwargs)

        for n in nodes:
            self.nodes.update({n.name: n})

    def initialize_baseline(self, **kwargs):
        """
        Set up node for baseline firing rates.
        Assumes the prior is on f * dt, where f is the baseline firing
        rate and dt is the time bin size. 
        """
        self._initialize_gaussian_nodes('baseline', (self.U,), (), (), 
            **kwargs)

        return self

    def initialize_fr_latents(self, **kwargs):
        """
        Set up node for firing rate effects due to latent variables.
        """
        self._initialize_gaussian_nodes('fr_latents', (self.U, self.K), 
            (self.K,), (), **kwargs)

        return self

    def initialize_fr_regressors(self, **kwargs):
        """
        Set up node for firing rate effects due to latent variables.
        """
        self._initialize_gaussian_nodes('fr_regressors', (self.U, self.R), 
            (self.R,), (), **kwargs)

        return self

    def initialize_overdispersion(self, **kwargs):
        """
        Set up trial-to-trial overdispersion on firing rates.
        """
        self._initialize_gaussian_nodes('overdispersion', (self.M,), 
            (), (), **kwargs)

        return self

    def initialize_latents(self, **kwargs):
        nodes = nd.initialize_HMM(self.K, 2, self.T, **kwargs)

        self.nodes.update({n.name: n for n in nodes})

        return self

    def expected_log_evidence(self):
        """
        Calculate E[log p(N, z|rest).
        """
        uu = self.Nframe['unit']
        nn = self.Nframe['count']
        tt = self.Nframe['time']

        Elogp = 0
        eta = 0

        if self.baseline:
            node = self.nodes['baseline']
            eta += node.expected_x()[uu]

        if self.latents:
            node = self.nodes['fr_latents']
            mu = node.expected_x()[uu]
            xi = self.nodes['HMM'].nodes['z'].z[1, tt]

            eta += np.sum(xi * mu, axis=1)

            # pieces for A and pi
            Elogp += self.nodes['HMM'].expected_log_state_sequence()

        if self.regressors:
            node = self.nodes['fr_regressors']
            mu = node.expected_x()[uu]
            xx = self.Xframe.values

            eta += np.sum(xx * mu, axis=1)

        if self.overdispersion:
            node = self.nodes['overdispersion']

        Elogp += np.sum(nn * eta - self.F())
        
        return Elogp

    def calc_log_evidence(self, idx):
        """
        Calculate p(N|z, rest) for use in updating HMM. Need only be
        correct up to an overall constant at each time (i.e., only
        relative probabilities at each time matter).
        """ 
        logpsi = np.empty((self.T, 2))

        # indices
        N = self.Nframe
        nn = N['count']
        uu = N['unit']

        # expectations
        bb = self.nodes['fr_latents']
        mu = bb.expected_x()[uu, idx]
        exp_bb = bb.expected_exp_x()[uu, idx]
        Fk = self.F(idx)

        N['logpsi0'] = -Fk
        N['logpsi1'] = nn * mu - exp_bb * Fk

        logpsi = N.groupby('time').sum()[['logpsi0', 'logpsi1']].values

        return logpsi 

    def F(self, k=None, update=False):
        """
        Accessor method to return the value of the F product.
        If k is specified, return F_{mu} (product over all but k), 
        else return F_{m} (product over all k). If update=True, 
        recalculate F before returning the result and cache the new 
        matrix. If flat=True, return the one-row-per-observation 
        version of F_{tu}.
        """

        if update:
            uu = self.Nframe['unit']
            tt = self.Nframe['time']

            if self.latents:
                xi = self.nodes['HMM'].nodes['z'].z[1]
                bb = self.nodes['fr_latents'].expected_exp_x()
                if k is not None:
                    zz = xi[tt, k]
                    w = bb[uu, k]
                    vv = ne.evaluate("1 - zz + zz * w")
                    self._Fpre[:, k] = vv
                else:
                    zz = xi[tt]
                    w = bb[uu]
                    vv = ne.evaluate("1 - zz + zz * w")
                    self._Fpre = vv
            else:
                self._Fpre = 1.

            # get other variables ready
            if self.baseline:
                lam = self.nodes['baseline'].expected_x()[uu]
                var_lam = self.nodes['baseline'].expected_var_x()[uu]
            else:
                lam = 0.
                var_lam = 0.

            if self.regressors:
                beta = self.nodes['fr_regressors'].expected_x()[uu]
                var_beta = self.nodes['fr_regressors'].expected_var_x()[uu]
                xbeta = np.sum(self.Xframe.values * beta, axis=1)
                xvb = np.sum(var_beta * self.Xframe.values ** 2, axis=1)
            else:
                xbeta = 0.
                xvb = 0.

            if self.overdispersion:
                od = self.nodes['overdispersion'].expected_x()
                var_od = self.nodes['overdispersion'].expected_var_x()
            else:
                od = 0.
                var_od = 0.

            rest = np.array(lam + xbeta + od + 0.5 * (var_lam + xvb + var_od))
            rr = rest.reshape(-1, 1)

            # work in log space to avoid over/underflow
            Fpre = self._Fpre
            dd = ne.evaluate("sum(log(Fpre), axis=1)")
            self._F_flat = ne.evaluate("exp(dd + rest)")
            ddd = dd[:, np.newaxis]
            self._Fk_flat = ne.evaluate("exp(ddd - log(Fpre) + rr)")

        if k is not None:
            return self._Fk_flat[..., k]
        else:
            return self._F_flat

    def update_baseline(self):
        node = self.nodes['baseline']
        mu = node.expected_x()
        var = node.expected_var_x()
        tau = node.prior_prec.expected_x()
        mm = node.prior_mean.expected_x()
        uu = self.Nframe['unit']
        nn = self.Nframe['count']

        # make an adjusted F that does not include our pars of interest
        F_adj = self.F() / node.expected_exp_x()[uu]

        # parameter of vectors to optimize over
        starts = np.concatenate((mu, np.log(var)))

        def minfun(x):
            jac = np.empty_like(x)
            mu = x[:self.U]
            kap = x[self.U:]
            var = np.exp(kap)
            bar_exp_eta = np.exp(mu + 0.5 * var)[uu] * F_adj

            elbo = -0.5 * np.sum(tau * (var + (mu - mm) ** 2))
            elbo += 0.5 * np.sum(np.log(var))
            elbo += np.sum(nn * mu[uu])
            elbo += -np.sum(bar_exp_eta)

            jac[:self.U] = -tau * (mu - mm) 
            jac[:self.U] += pd.DataFrame(nn - bar_exp_eta).groupby(uu).sum().values.squeeze()
            jac[self.U:] = -0.5 * tau * var + 0.5 
            jac[self.U:] += -0.5 * pd.DataFrame(var[uu] * bar_exp_eta).groupby(uu).sum().values.squeeze()

            return np.log(-elbo), jac / elbo

        res = minimize(minfun, starts, jac=True, 
            options={'maxiter': self.maxiter})

        node.post_mean = res.x[:self.U]
        node.post_prec = np.exp(-res.x[self.U:])
        self.F(update=True)

    def update_fr_latents(self, idx):
        node = self.nodes['fr_latents']
        mu = node.expected_x()[..., idx]
        var = node.expected_var_x()[..., idx]
        tau = node.prior_prec.expected_x()[..., idx]
        mm = node.prior_mean.expected_x()[..., idx]
        uu = self.Nframe['unit']
        nn = self.Nframe['count']
        tt = self.Nframe['time']

        # make an adjusted F that does not include our pars of interest
        F_adj = self.F(idx)
        xi = self.nodes['HMM'].nodes['z'].z[1, tt, idx]

        # parameter of vectors to optimize over
        starts = np.concatenate((mu, np.log(var)))

        def minfun(x):
            jac = np.empty_like(x)
            mu = x[:self.U]
            kap = x[self.U:]
            var = np.exp(kap)
            bar_exp_eta = (1 - xi + xi * np.exp(mu + 0.5 * var)[uu]) * F_adj
            F_tilde = np.exp(mu + 0.5 * var)[uu] * F_adj

            elbo = -0.5 * np.sum(tau * (var + (mu - mm) ** 2))
            elbo += 0.5 * np.sum(np.log(var))
            elbo += np.sum(nn * mu[uu] * xi)
            elbo += -np.sum(bar_exp_eta)

            jac[:self.U] = -tau * (mu - mm) 
            jac[:self.U] += pd.DataFrame((nn - F_tilde) * xi).groupby(uu).sum().values.squeeze()
            jac[self.U:] = -0.5 * tau * var + 0.5 
            jac[self.U:] += -0.5 * pd.DataFrame(xi * var[uu] * F_tilde).groupby(uu).sum().values.squeeze()

            return np.log(-elbo), jac / elbo

        res = minimize(minfun, starts, jac=True, 
            options={'maxiter': self.maxiter})

        node.post_mean[..., idx] = res.x[:self.U]
        node.post_prec[..., idx] = np.exp(-res.x[self.U:])
        self.F(idx, update=True)

    def update_fr_regressors(self, idx):
        node = self.nodes['fr_regressors']
        mu = node.expected_x()[..., idx]
        var = node.expected_var_x()[..., idx]
        tau = node.prior_prec.expected_x()[..., idx]
        mm = node.prior_mean.expected_x()[..., idx]
        uu = self.Nframe['unit']
        nn = self.Nframe['count']

        # make an adjusted F that does not include our pars of interest
        xx = self.Xframe.values[..., idx]
        xx2 = xx ** 2
        F_adj = self.F() / np.exp(mu[uu] * xx + 0.5 * var[uu] * xx2)

        # parameter of vectors to optimize over
        starts = np.concatenate((mu, np.log(var)))

        def minfun(x):
            jac = np.empty_like(x)
            mu = x[:self.U]
            kap = x[self.U:]
            var = np.exp(kap)
            bar_exp_eta = np.exp(mu[uu] * xx + 0.5 * var[uu] * xx2) * F_adj

            elbo = -0.5 * np.sum(tau * (var + (mu - mm) ** 2))
            elbo += 0.5 * np.sum(np.log(var))
            elbo += np.sum(nn * mu[uu] * xx)
            elbo += -np.sum(bar_exp_eta)

            jac[:self.U] = -tau * (mu - mm) 
            jac[:self.U] += pd.DataFrame((nn - bar_exp_eta) * xx).groupby(uu).sum().values.squeeze()
            jac[self.U:] = -0.5 * tau * var + 0.5
            jac[self.U:] += -0.5 * pd.DataFrame(var[uu] * bar_exp_eta * xx2).groupby(uu).sum().values.squeeze()

            return np.log(-elbo), jac / elbo

        res = minimize(minfun, starts, jac=True, 
            options={'maxiter': self.maxiter})

        node.post_mean[..., idx] = res.x[:self.U]
        node.post_prec[..., idx] = np.exp(-res.x[self.U:])
        self.F(update=True)

    def update_overdispersion(self):
        node = self.nodes['overdispersion']
        mu = node.expected_x()
        var = node.expected_var_x()
        tau = node.prior_prec.expected_x()
        mm = node.prior_mean.expected_x()
        nn = self.Nframe['count']

        # make an adjusted F that does not include our pars of interest
        F_adj = self.F() / node.expected_exp_x()

        def objfun(x):
            mu = x[:self.M]
            kap = x[self.M:]
            var = np.exp(kap)
            bar_exp_eta = np.exp(mu + 0.5 * var) * F_adj

            elbo = -0.5 * np.sum(tau * (var + (mu - mm) ** 2))
            elbo += 0.5 * np.sum(np.log(var))
            elbo += np.sum(nn * mu)
            elbo += -np.sum(bar_exp_eta)

            return -elbo

        def gradfun(x):
            jac = np.empty_like(x)
            mu = x[:self.M]
            kap = x[self.M:]
            var = np.exp(kap)
            bar_exp_eta = np.exp(mu + 0.5 * var) * F_adj

            jac[:self.M] = -tau * (mu - mm) 
            jac[:self.M] += (nn - bar_exp_eta)
            jac[self.M:] = -0.5 * tau * var + 0.5 
            jac[self.M:] += -0.5 * var * bar_exp_eta

            return -jac

        # parameter of vectors to optimize over
        starts = np.concatenate((mu, np.log(var)))
        start_g = gradfun(starts)

        alpha = line_search(objfun, gradfun, starts, -start_g, gfk=start_g)
        if alpha[0] is not None:
            xnew = starts - alpha[0] * start_g

            node.post_mean = xnew[:self.M]
            node.post_prec = np.exp(-xnew[self.M:])
            self.F(update=True)


    def finalize(self):
        """
        This should be called once all the relevant variables are initialized.
        """
        if 'baseline' in self.nodes:
            self.baseline = True
            self.nodes['baseline'].update = self.update_baseline
        else:
            self.baseline = False

        if {'HMM', 'fr_latents'}.issubset(self.nodes):
            self.latents = True
            self.nodes['fr_latents'].update = self.update_fr_latents

            self.nodes['HMM'].update_finalizer = (
                lambda idx: self.F(idx, update=True))
        else:
            self.latents = False

        if 'fr_regressors' in self.nodes:
            self.regressors = True
            self.nodes['fr_regressors'].update = self.update_fr_regressors
        else:
            self.regressors = False

        if 'overdispersion' in self.nodes:
            self.overdispersion = True
            self.nodes['overdispersion'].update = self.update_overdispersion
        else:
            self.overdispersion = False

        self.F(update=True)
        return self

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
            print "Elogp = {}".format(Elogp)

        for _, node in self.nodes.iteritems():
            logp = node.expected_log_prior()
            logq = node.entropy()
            if print_pieces:
                print "{}: Elogp = {}, H = {}".format(node.name, logp, logq)

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
                assert(Lval >= lastL)
                lastL = Lval
            if doprint:
                print "         updated baselines: L = {}".format(Lval)

        if self.latents:
            for k in xrange(self.K):
                self.nodes['fr_latents'].update(k)
                if self.nodes['fr_latents'].has_parents:
                    self.nodes['fr_latents'].update_parents(k)
                if calc_L:
                    Lval = self.L(keeplog=keeplog, print_pieces=print_pieces) 
                    assert(Lval >= lastL)
                    lastL = Lval
                if doprint:
                    print ("chain {}: updated firing rate effects: L = {}"
                        ).format(k, Lval)

        if self.regressors:
            for k in xrange(self.R):
                self.nodes['fr_regressors'].update(k)
                if self.nodes['fr_regressors'].has_parents:
                    self.nodes['fr_regressors'].update_parents(k)
                if calc_L:
                    Lval = self.L(keeplog=keeplog, print_pieces=print_pieces) 
                    assert(Lval >= lastL)
                    lastL = Lval
                if doprint:
                    print ("regressor {}: updated firing rate effects: L = {}"
                        ).format(k, Lval)

        if self.overdispersion:
            self.nodes['overdispersion'].update()
            if self.nodes['overdispersion'].has_parents:
                self.nodes['overdispersion'].update_parents()
            if calc_L:
                Lval = self.L(keeplog=keeplog, print_pieces=print_pieces) 
                assert(Lval >= lastL)
                lastL = Lval
            if doprint:
                print ("         updated overdispersion effects: L = {}"
                    ).format(Lval)

        # E step        
        if self.latents:
            for k in xrange(self.K):
                logpsi = self.calc_log_evidence(k)
                self.nodes['HMM'].update(k, logpsi)
                if calc_L:
                    Lval = self.L(keeplog=keeplog, print_pieces=print_pieces) 
                    assert(Lval >= lastL)
                    lastL = Lval
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
