import numpy as np
from scipy.special import digamma, gammaln
from ConstNode import ConstNode

class GammaNode:
    """
    Gamma distribution. Uses (shape, rate) parameterization.
    """
    def __init__(self, prior_shape, prior_rate, post_shape,
        post_rate, name='gamma'):
        if prior_shape.shape != prior_rate.shape:
            raise ValueError('Dimensions of priors must agree!')
        if post_shape.shape != post_rate.shape:
            raise ValueError('Dimensions of posteriors must agree!')
        try: 
            np.broadcast(post_shape, prior_shape)
        except:
            raise ValueError('Dimensions of prior and posterior could not be broadcast together.')

        # if any parameters are passed as arrays, wrap in 
        # constant node
        if isinstance(prior_shape, np.ndarray):
            self.prior_shape = prior_shape.view(ConstNode)
        else:
            self.prior_shape = prior_shape

        if isinstance(prior_rate, np.ndarray):
            self.prior_rate = prior_rate.view(ConstNode)
        else:
            self.prior_rate = prior_rate

        self.post_shape = post_shape
        self.post_rate = post_rate
        self.name = name
        self.shape = prior_shape.shape

    def expected_x(self):
        return self.post_shape / self.post_rate

    def expected_log_x(self):
        return digamma(self.post_shape) - np.log(self.post_rate)

    def expected_log_prior(self):
        """
        Calculate expected value of log prior under the posterior distribution.
        """
        alpha = self.prior_shape.expected_x()
        beta = self.prior_rate.expected_x()
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

