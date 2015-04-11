import numpy as np
from scipy.special import digamma, gammaln

class GammaNode:
    """
    Gamma distribution. Uses (shape, rate) parameterization.
    """
    def __init__(self, prior_shape, prior_rate, post_shape,
        post_rate, name='gamma'):
        if prior_shape.shape != prior_rate.shape:
            raise ValueError('Dimensions of priors must agree!')
        if post_shape.shape != post_rate.shape:
            raise ValueError('Dimensions of priors must agree!')

        self.prior_shape = prior_shape
        self.prior_rate = prior_rate
        self.post_shape = post_shape
        self.post_rate = post_rate
        self.name = name

    def expected_x(self):
        return self.post_shape / self.post_rate

    def expected_log_x(self):
        return digamma(self.post_shape) - np.log(self.post_rate)

    def expected_log_prior(self):
        """
        Calculate expected value of log prior under the posterior distribution.
        """
        alpha = self.prior_shape
        beta = self.prior_rate
        elp = (alpha - 1) * self.expected_log_x() 
        elp += -beta * self.expected_x() 
        elp += alpha * np.log(beta) 
        elp += -gammaln(alpha)

        return np.sum(elp)

    def entropy(self):
        """
        Calculate differential entropy of posterior.
        """
        alpha = self.post_shape
        beta = self.post_rate
        H = alpha - np.log(beta)
        H += gammaln(alpha)
        H += (1 - alpha) * digamma(alpha)

        return np.sum(H)

