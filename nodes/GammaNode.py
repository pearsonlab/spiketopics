import numpy as np
from scipy.special import digamma, gammaln

class GammaNode:
    """
    Gamma distribution. Uses (shape, rate) parameterization.
    """
    def __init__(self, prior, post, name='gamma'):
        if prior.shape[-1] != 2:
            raise ValueError('Last dimension of prior parameters array must be 2')  
        if post.shape[-1] != 2:
            raise ValueError('Last dimension of posterior parameters array must be 2')  

        self.prior_shape = prior[..., 0]
        self.prior_rate = prior[..., 1]
        self.post_shape = post[..., 0]
        self.post_rate = post[..., 1]

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