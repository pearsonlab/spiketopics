from __future__ import division
import numpy as np
from scipy.special import digamma, gammaln
from utility_nodes import ConstNode

class BernoulliNode:
    """
    Bernoulli distribution. May be an array.
    """
    def __init__(self, prior, post, name='dirichlet'):
        if isinstance(prior, np.ndarray):
            self.prior = prior.copy().view(ConstNode)
        else:
            self.prior = prior

        self.post = post.copy()
        self.name = name
        self.shape = post.shape

    def expected_x(self):
        """
        Calculate expected value of the transition matrix values under
        the posterior distribution.
        """
        return self.post

    def expected_log_prior(self):
        """
        Calculate expected value of log prior under the posterior distribution.
        """
        p = self.prior.expected_x()
        pp = self.expected_x()
        elp = pp * np.log(p[0]) + (1 - pp) * np.log(p[1])
        elp = elp.view(np.ndarray)
        elp[np.isnan(elp)] = 0

        return np.sum(elp)

    def entropy(self):
        """
        Calculate differential entropy of posterior.
        """
        p = self.post
        H = -p * np.log(p) - (1 - p) * np.log1p(-p)
        H[np.isnan(H)] = 0

        return np.sum(H)

    def update(self, idx, ess):
        """
        Update posterior given expected sufficient statistics.
        """
        self.post[..., idx] = self.prior[..., idx] + ess

        return self
