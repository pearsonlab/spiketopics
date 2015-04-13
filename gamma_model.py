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

    def initialize_baseline(self, prior_shape, prior_rate, 
        post_shape, post_rate):
        """
        Set up node for baseline firing rate effects.
        Assumes the prior is on f * dt, where f is the baseline firing
        rate and dt is the time bin size.
        """
        if prior_shape.shape != (self.U,):
            raise ValueError('Prior has shape inconsistent with data.')

        self.baseline = nd.GammaNode(prior_shape, prior_rate, post_shape, 
            post_rate)

        self.Lterms.append(self.baseline)

        return self



