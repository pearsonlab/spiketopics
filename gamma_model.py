from __future__ import division
import numpy as np
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

        self.log = {'L': [], 'H': []}  # for debugging
        self.Lvalues = []  # for recording optimization objective

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
        self._initialize_gamma_nodes('baseline', (self.U,), (1,), **kwargs)

        return self

    def initialize_fr_latents(self, **kwargs):
        """
        Set up node for firing rate effects due to latent variables.
        """
        self._initialize_gamma_nodes('fr_latents', (self.K, self.U), 
            (self.K,), **kwargs)

        return self

    def initialize_fr_regressors(self, **kwargs):
        """
        Set up node for firing rate effects due to latent variables.
        """
        self._initialize_gamma_nodes('fr_regressors', (self.R, self.U), 
            (self.R,), **kwargs)

        return self

    def initialize_overdispersion(self, **kwargs):
        """
        Set up trial-to-trial overdispersion on firing rates.
        """
        self._initialize_gamma_nodes('overdispersion', (self.M,), 
            (1,), **kwargs)

        return self

    def initialize_latents(self, **kwargs):
        node = nd.initialize_HMM(self.K, 2, self.T, **kwargs)

        self.nodes.update({node.name: node})

        return self

    def finalize(self):
        """
        This should be called once all the relevant variables are initialized.
        """
        if {'HMM', 'fr_latents'}.issubset(self.nodes):
            self.F_prod(update=True)
        if 'fr_regressors' in self.nodes:
            self.G_prod(update=True)

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
            xi = self.nodes['HMM'].nodes['z'].z[1]
            lam = self.nodes['fr_latents'].expected_x()
            if k is not None:
                zz = xi[:, k, np.newaxis]
                w = lam[k]
                vv = ne.evaluate("1 - zz + zz * w")
                self._Fpre[:, k, :] = vv
            else:
                zz = xi[:, :, np.newaxis]
                w = lam
                vv = ne.evaluate("1 - zz + zz * w")
                self._Fpre = vv

            # work in log space to avoid over/underflow
            uu = self.Nframe['unit']
            tt = self.Nframe['time']
            Fpre = self._Fpre
            dd = ne.evaluate("sum(log(Fpre), axis=1)")
            self._Ftu = ne.evaluate("exp(dd)")
            ddd = dd[:, np.newaxis, :]
            self._Ftku = ne.evaluate("exp(ddd - log(Fpre))")
            self._Fflat = self._Ftu[tt, uu]

        if k is not None:
            return self._Ftku[:, k, :]
        elif flat:
            return self._Fflat
        else:
            return self._Ftu

    def G_prod(self, k=None, update=False):
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
            lam = self.nodes['fr_regressors'].expected_x()
            uu = self.Nframe['unit']
            if k is not None:
                zz = lam[k][uu]
                # get x values for kth regressor
                xx = self.Xframe.values[:, k]
                vv = ne.evaluate("zz ** xx")
                self._Gpre[:, k] = vv
            else:
                zz = lam[:, uu].T
                # get x values for kth regressor; col 0 = time
                xx = self.Xframe.values
                vv = ne.evaluate("zz ** xx")
                self._Gpre = vv

            # work in log space to avoid over/underflow
            Gpre = self._Gpre
            dd = ne.evaluate("sum(log(Gpre), axis=1)")
            self._Gtu = ne.evaluate("exp(dd)")
            ddd = dd[:, np.newaxis]
            self._Gtku = ne.evaluate("exp(ddd - log(Gpre))")

        if k is not None:
            return self._Gtku[:, k]
        else:
            return self._Gtu


