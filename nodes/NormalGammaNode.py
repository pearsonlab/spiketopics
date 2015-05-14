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
        return ((1. / self.post_scaling) + 
            self.post_mean**2 * (self.post_shape / self.post_rate))

    def expected_log_prior(self):
        """
        Calculate expected value of log prior under the posterior distribution.
        """
        alpha = self.prior_shape
        beta = self.prior_rate
        mu = self.prior_mean
        lam = self.prior_scaling
        elp = (alpha - 0.5) * self.expected_log_t()
        elp += -beta * self.expected_t()
        elp += -0.5 * lam * self.expected_txx()
        elp += lam * mu * self.expected_tx()
        elp += -0.5 * lam * self.expected_t() * mu ** 2
        elp += alpha * np.log(beta) 
        elp += -gammaln(alpha)
        elp += 0.5 * np.log(lam / (2 * np.pi))
        elp = elp.view(np.ndarray)

        return np.sum(elp)

    def entropy(self):
        """
        Calculate differential entropy of posterior.
        """
        lam = self.post_scaling
        alpha = self.post_shape
        beta = self.post_rate
        H = alpha - np.log(beta) + gammaln(alpha)
        H += (1 - alpha) * digamma(alpha)
        H += -0.5 * np.log(lam / (2 * np.pi))
        H += -0.5 * (digamma(alpha) - np.log(beta))
        H += 0.5 

        return np.sum(H)

    def update(self, idx, ess_1, ess_2, ess_3, ess_4):
        """
        Update posterior given sufficient statistics for the natural 
        parameters.
        """
        # calculate prior values in natural parameters
        prior_1 = self.prior_shape - 0.5
        prior_2 = (-self.prior_rate - 0.5 * self.prior_scaling * 
            self.prior_mean**2)
        prior_3 = self.prior_scaling * self.prior_mean
        prior_4 = -0.5 * self.prior_scaling

        # update to post using ess
        post_1 = ess_1 + prior_1[..., idx]
        post_2 = ess_2 + prior_2[..., idx]
        post_3 = ess_3 + prior_3[..., idx]
        post_4 = ess_4 + prior_4[..., idx]

        # convert back
        self.post_shape[..., idx] = post_1 + 0.5
        self.post_rate[..., idx] = -post_2 + 0.25 * (post_3**2 / post_4)
        self.post_mean[..., idx] = -0.5 * post_3 / post_4
        self.post_scaling[..., idx] = -2 * post_4








