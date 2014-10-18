"""
Tests for Gamma-Poisson model.
"""
import numpy as np
import scipy.stats as stats

class Forwards_Backwards_Integration:
    def setup_class():
        """
        Make some sample Markov chain data to run inference on.
        """

        np.random.seed(12345)

        U = 100  # units
        T = 5000  # time points/frames
        Kdata = 5  # categories
        dt = 1 / 30  # frames per second

        # transition probabilities
        v0_a = 1
        v0_b = 200
        v1_a = 100
        v1_b = 1

        # make vector of transition probabilities
        # each row is a category, each column the transition
        # probability from that state to state 1
        # (this is easy to do without matrices since we have
        # only two states)
        v0 = stats.beta.rvs(v0_a, v0_b, size=Kdata)
        v1 = stats.beta.rvs(v1_a, v1_b, size=Kdata)
        trans_probs = np.vstack([v0, v1]).T 

        # pick initial states for each category
        z0_probs = stats.beta.rvs(50, 50, size=Kdata)
        z0 = stats.bernoulli.rvs(z0_probs)

        # make the chain by evolving each category through time
        chain = np.empty((Kdata, T), dtype='int')

        # initialize
        chain[:, 0] = z0

        for t in xrange(1, T):
            p_trans_to_1 = trans_probs[range(Kdata), chain[:, t - 1]]
            chain[:, t] = stats.bernoulli.rvs(p_trans_to_1)
            
        # include a baseline regressor that is always turned on
        chain[0, :] = 1

        # set up firing rates for each category, unit
        aa = 3, 1
        bb = 2, 1
        lam = stats.gamma.rvs(a=aa[1], scale=bb[1], size=(Kdata, U))
        lam[0, :] = stats.gamma.rvs(a=aa[0], scale=bb[0], size=U)  # baselines 

        # calculate rate for each time
        fr = np.exp(chain.T.dot(np.log(lam))) * dt
        fr = fr + 1e-5  # in case we get exactly 0

        # draw from Poisson
        N = stats.poisson.rvs(fr)

    def test_stuff():
        pass