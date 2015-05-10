from __future__ import division
import numpy as np
from scipy.special import digamma, gammaln
from utility_nodes import ConstNode

class NormalGammaNode:
    """
    Normal-Gamma distribution for (mean, precision). Parameters are 
    (mean, precision scaling, shape, rate).
    """
    def __init__(self, prior_mean, prior_scaling, prior_shape, prior_rate,
        post_mean, post_scaling, post_shape, post_rate, name='normalgamma'):
        if prior_scaling.shape != prior_mean.shape:
            raise ValueError('Dimensions of priors must agree!')
        if prior_shape.shape != prior_mean.shape:
            raise ValueError('Dimensions of priors must agree!')
        if prior_rate.shape != prior_mean.shape:
            raise ValueError('Dimensions of priors must agree!')
        if prior_shape.shape != prior_mean.shape:
            raise ValueError('Dimensions of priors must agree!')
        if post_scaling.shape != post_mean.shape:
            raise ValueError('Dimensions of posteriors must agree!')
        if post_shape.shape != post_mean.shape:
            raise ValueError('Dimensions of posteriors must agree!')
        if post_rate.shape != post_mean.shape:
            raise ValueError('Dimensions of posteriors must agree!')
        if post_shape.shape != post_mean.shape:
            raise ValueError('Dimensions of posteriors must agree!')

        # if any parameters are passed as arrays, wrap in 
        # constant node
        if isinstance(prior_mean, np.ndarray):
            self.prior_mean = prior_mean.copy().view(ConstNode)
        else:
            self.prior_mean = prior_mean

        if isinstance(prior_scaling, np.ndarray):
            self.prior_scaling = prior_scaling.copy().view(ConstNode)
        else:
            self.prior_scaling = prior_scaling
        if isinstance(prior_shape, np.ndarray):
            self.prior_shape = prior_shape.copy().view(ConstNode)
        else:
            self.prior_shape = prior_shape

        if isinstance(prior_rate, np.ndarray):
            self.prior_rate = prior_rate.copy().view(ConstNode)
        else:
            self.prior_rate = prior_rate

        self.post_mean = post_mean.copy()
        self.post_scaling = post_scaling.copy()
        self.post_shape = post_shape.copy()
        self.post_rate = post_rate.copy()
        self.name = name
        self.shape = post_mean.shape

    def expected_t(self):
        return self.post_shape / self.post_rate

    def expected_log_t(self):
        return digamma(self.post_shape) - np.log(self.post_rate)

    def expected_tx(self):
        return self.post_mean * (self.post_shape / self.post_rate)

    def expected_txx(self):
        return (1. / self.post_scaling + 
            self.post_mean ** 2 * (self.post_shape / self.post_rate))

    def expected_log_prior(self):
        """
        Calculate expected value of log prior under the posterior distribution.
        """
        alpha = self.prior_shape.expected_x()
        beta = self.prior_rate.expected_x()
        mu = self.prior_mean.expected_x()
        lam = self.prior_scaling.expected_x()
        elp = (alpha - 0.5) * self.expected_log_t()
        elp += -beta * self.expected_t()
        elp += -0.5 * lam * self.expected_txx()
        elp += lam * mu * self.expected_tx()
        elp += -0.5 * lam * mu ** 2
        elp += alpha * np.log(beta) 
        elp += 0.5 * np.log(lam) - 0.5 * np.log(2 * np.pi)
        elp += -gammaln(alpha)
        elp = elp.view(np.ndarray)

    def entropy(self):
        """
        Calculate differential entropy of posterior.
        """
        mu = self.post_mean
        lam = self.post_scaling
        alpha = self.post_shape
        beta = self.post_rate
        H = alpha - np.log(beta)
        H += gammaln(alpha)
        H += (0.5 - alpha) * digamma(alpha)
        H += 0.5 * np.log(2 * np.pi * np.e / lam)
        H += 0.5 * lam * (mu ** 2) * (1 - (alpha / beta))

        return np.sum(H)

    def update(self, ess_mean, ess_scaling, ess_shape, ess_rate):
        """
        Update posterior given expected sufficient statistics.
        """
        self.post_mean = self.prior_mean + ess_mean
        self.post_scaling = self.prior_scaling + ess_scaling
        self.post_shape = self.prior_shape + ess_shape
        self.post_rate = self.prior_rate + ess_rate

        return self








