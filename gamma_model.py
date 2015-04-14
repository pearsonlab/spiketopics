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

    def _initialize_gamma_hierarchy(self, basename, parent_shape, 
        child_shape, **kwargs):
        """
        Initialize a hierarchical gamma variable: 
            lambda ~ Gamma(c, c * m)
            c ~ Gamma
            m ~ Gamma

        basename is the name of the name of the child variable 
        parent_shape and child_shape are the shapes of the parent and
            child variables
        """
        par_shapes = ({'prior_shape_shape': parent_shape, 
            'prior_shape_rate': parent_shape, 
            'prior_mean_shape': parent_shape, 'prior_mean_rate': parent_shape, 
            'post_shape_shape': parent_shape, 'post_shape_rate': parent_shape, 
            'post_mean_shape': parent_shape, 'post_mean_rate': parent_shape, 
            'post_child_shape': child_shape, 'post_child_rate': child_shape})

        # error checking
        for var, shape in par_shapes.iteritems():
            if var not in kwargs:
                raise ValueError('Argument missing: {}'.format(var))
            elif kwargs[var].shape != shape:
                raise ValueError('Argument has wrong shape: {}'.format(var))

        shape = nd.GammaNode(kwargs['prior_shape_shape'], 
            kwargs['prior_shape_rate'], kwargs['post_shape_shape'], 
            kwargs['post_shape_rate'])

        mean = nd.GammaNode(kwargs['prior_mean_shape'], 
            kwargs['prior_mean_rate'], kwargs['post_mean_shape'], 
            kwargs['post_mean_rate'])

        child = nd.GammaNode(shape, nd.ProductNode(shape, mean),
            kwargs['post_child_shape'], kwargs['post_child_rate'])

        setattr(self, basename + '_shape', shape)
        setattr(self, basename + '_mean', mean)
        setattr(self, basename, child)

        self.Lterms.extend([child, shape, mean])

        return self

    def _initialize_gamma(self, name, node_shape, **kwargs):
        """
        Initialize a gamma variable.
        """
        par_shapes = ({'prior_shape': node_shape, 'prior_rate': node_shape,
            'post_shape': node_shape, 'post_rate': node_shape })

        # error checking
        for var, shape in par_shapes.iteritems():
            if var not in kwargs:
                raise ValueError('Argument missing: {}'.format(var))
            elif kwargs[var].shape != shape:
                raise ValueError('Argument has wrong shape: {}'.format(var))

        node = nd.GammaNode(kwargs['prior_shape'], kwargs['prior_rate'], 
            kwargs['post_shape'], kwargs['post_rate'])

        setattr(self, name, node)

        self.Lterms.append(node)

        return self

    def initialize_baseline(self, **kwargs):
        """
        Set up node for baseline firing rates.
        Assumes the prior is on f * dt, where f is the baseline firing
        rate and dt is the time bin size. 
        """
        node_shape = (self.U,)

        self._initialize_gamma('baseline', node_shape, **kwargs)

        return self

    def initialize_baseline_hierarchy(self, **kwargs):
        """
        Initialize baseline with hierarchy over units.
        """
        child_shape = (self.U,)
        parent_shape = (1,)

        self._initialize_gamma_hierarchy('baseline', parent_shape,
            child_shape, **kwargs)

        return self

    def initialize_fr_latents(self, **kwargs):
        """
        Set up node for firing rate effects due to latent variables.
        """
        node_shape = (self.K, self.U)

        self._initialize_gamma('fr_latents', node_shape, **kwargs)

        return self

    def initialize_fr_latents_hierarchy(self, **kwargs):
        """
        Initialize firing rate effects for latent states with 
        hierarchy over units.
        """
        child_shape = (self.K, self.U)
        parent_shape = (self.K,)

        self._initialize_gamma_hierarchy('fr_latents', parent_shape,
            child_shape, **kwargs)

        return self
