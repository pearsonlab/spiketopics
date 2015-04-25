from __future__ import division
import numpy as np
from utility_nodes import ConstNode

class GaussianNode:
    """
    Normal distribution. Uses (mean, precision) parameterization.
    """
    def __init__(self, prior_mean, prior_prec, post_mean,
        post_prec, name='guassian'):
        if prior_mean.shape != prior_prec.shape:
            raise ValueError('Dimensions of priors must agree!')
        if post_mean.shape != post_prec.shape:
            raise ValueError('Dimensions of posteriors must agree!')

        # if any parameters are passed as arrays, wrap in 
        # constant node
        if isinstance(prior_mean, np.ndarray):
            self.prior_mean = prior_mean.copy().view(ConstNode)
        else:
            self.prior_mean = prior_mean

        if isinstance(prior_prec, np.ndarray):
            self.prior_prec = prior_prec.copy().view(ConstNode)
        else:
            self.prior_prec = prior_prec

        self.post_mean = post_mean.copy()
        self.post_prec = post_prec.copy()
        self.name = name
        self.shape = prior_mean.shape

    def expected_x(self):
        return self.post_mean 

    def expected_var_x(self):
        return 1. / self.post_prec 

    def expected_prec_x(self):
        return self.post_prec 

    def expected_log_prior(self):
        """
        Calculate expected value of log prior under the posterior distribution.
        """
        mu = self.prior_mean.expected_x()
        tau = self.prior_prec.expected_x()
        log_tau = self.prior_prec.expected_log_x()
        elp = -0.5 * tau * self.expected_var_x() 
        elp += -0.5 * (self.expected_x() - mu) ** 2 
        elp += -0.5 * np.log(2 * np.pi) + 0.5 * log_tau
        elp = elp.view(np.ndarray)

        return np.sum(elp)

    def entropy(self):
        """
        Calculate differential entropy of posterior.
        """
        tau = self.post_prec 
        H = 0.5 * (1 + np.log(2 * np.pi) - np.log(tau))

        return np.sum(H)

    def update(self, ess_shape, ess_rate):
        """
        Update posterior given expected sufficient statistics.
        """
        self.post_shape = self.prior_shape + ess_shape
        self.post_rate = self.prior_rate + ess_rate

        return self
