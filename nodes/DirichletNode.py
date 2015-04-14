import numpy as np
from scipy.special import digamma, gammaln
from utility_nodes import ConstNode

class DirichletNode:
    """
    Dirichlet distribution. May be an array. Axis 0 is assumed to be
    the probability dimension.
    """
    def __init__(self, prior, post, name='dirichlet'):
        try: 
            np.broadcast(prior, post)
        except:
            raise ValueError('Dimensions of prior and posterior could not be broadcast together.')

        self.M = prior.shape[0]  # number of states

        if isinstance(prior, np.ndarray):
            self.prior = prior.view(ConstNode)
        else:
            self.prior = prior

        self.post = post
        self.name = name
        self.shape = prior.shape

    def expected_x(self):
        """
        Calculate expected value of the transition matrix values under
        the posterior distribution.
        """
        return self.post / np.sum(self.post, axis=0, keepdims=True)

    def expected_log_x(self):
        """
        Calculate expected log value of the transition matrix values under
        the posterior distribution.
        """
        return digamma(self.post) / digamma(np.sum(self.post, axis=0, keepdims=True))

    @staticmethod
    def logB(alpha):
        """
        logB = sum_i log Gamma(alpha_i) - log Gamma(sum_i alpha_i)
        """
        return np.sum(gammaln(alpha), axis=0) - gammaln(np.sum(alpha, axis=0))

    def expected_log_prior(self):
        """
        Calculate expected value of log prior under the posterior distribution.
        """
        alpha = self.prior
        elp = np.sum((alpha - 1) * self.expected_x(), axis=0)
        elp += -self.logB(alpha)

        return np.sum(elp)

    def entropy(self):
        """
        Calculate differential entropy of posterior.
        """
        alpha = self.post
        H = self.logB(alpha)
        H += (alpha[0] - self.M) * digamma(alpha[0])
        H += -np.sum((alpha[1:] - 1) * digamma(alpha[1:]), axis=0)

        return H

    def update(self, ess):
        """
        Update posterior given expected sufficient statistics.
        """
        self.post = self.prior + ess

        return self
