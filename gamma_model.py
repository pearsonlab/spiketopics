from __future__ import division
import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln, betaln
from scipy.optimize import minimize
from forward_backward import fb_infer
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

        self.log = {'L': [], 'H': []}  # for debugging
        self.Lvalues = []  # for recording optimization objective
        self.Lterms = []  # holds piece of optimization objective

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
            setattr(self, n.name, n)

        self.Lterms.extend(nodes)

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
