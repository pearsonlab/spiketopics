"""
Node definining a segmentation of time into a sequence of Bernoulli posteriors.
"""
from __future__ import division
import numpy as np

from ..pelt import find_changepoints, calc_state_probs
from scipy.stats import bernoulli
from scipy.special import logit

class ZNode:
    """
    Distribution for a segmentation of time into a sequence of states, each
    described by a Bernoulli posterior. The locations of the changepoints are
    calculated via the PELT algorithm.

    As with MarkovChainNode, this is only a wrapper for the values z.
    """
    def __init__(self, z_init, theta, alpha, name='znode'):
        if len(z_init.shape) < 2:
            raise ValueError(
                'State variable must have at least two dimensions')

        self.M = z_init.shape[0]
        self.T = z_init.shape[1]
        if len(z_init.shape) > 2:
            self.K = z_init.shape[2:]
        else:
            self.K = (1,)
        self.theta = theta
        self.alpha = alpha

        self.name = name
        self.shape = z_init.shape

        # set inits (but normalize)
        self.z = z_init / np.sum(z_init, axis=0, keepdims=True)

    def update(self, idx, new_z):
        self.z[:, :, idx] = new_z

        return self

class SegmentNode:
    """
    Node representing a changepoint segmentation model. Comprises a ZNode
    containing the expected value of the state in each segment.
    """
    def __init__(self, z, name='segmodel'):
        M = z.shape[0]
        T = z.shape[1]
        if len(z.shape) > 2:
            K = z.shape[2:]
        else:
            K = (1,)

        self.M = M
        self.T = T
        self.K = K
        self.shape = z.shape
        self.name = name
        self.update_finalizer = None

        self.nodes = {'z': z} 

        self.Hz = np.zeros(K)
        self.elp = np.zeros(K)

    def update(self, idx, log_evidence):
        """
        Update the chain with index idx.
        idx needs to be an object capable of indexing the relevant Markov
            Chain (i.e., z after the first two (M, T) indices, A after the
            first two (M, M) indices, etc.)
        log_evidence is log p(y|z, theta) (T, M) where y is the 
            observed data, z is the latent state variable(s), 
            and theta are the other parameters of the model.
        """

        ########### update chains
        # (T, M)
        psi = log_evidence #- np.amax(log_evidence, axis=1, keepdims=True) 
        xi = np.zeros_like(psi)
        theta = self.nodes['z'].theta
        alpha = self.nodes['z'].alpha

        cplist = find_changepoints(psi, theta, alpha)
        xi[:, 1] = calc_state_probs(psi, theta, cplist)
        xi[:, 0] = 1 - xi[:, 1]
        run_starts = [t + 1 for t in cplist]
        Ez = xi[run_starts, 1]  # posterior probabilities in each segment

        xi = xi.T  # now (M, T)
        self.nodes['z'].update(idx, xi) 

        ########### calculate entropy pieces
        self.Hz[idx] = np.sum(bernoulli.entropy(Ez))

        assert(np.all(self.Hz >= 0))

        ########### calculate expected log prior
        pi = np.sum(Ez)
        m = len(Ez)
        print pi, m
        elp = pi * logit(theta) - m * (alpha - np.log1p(-theta))
        self.elp[idx] = elp
        print self.elp

        if self.update_finalizer is not None:
            self.update_finalizer(idx)

        return self

    def entropy(self):
        return np.sum(self.Hz) 

    def expected_log_prior(self):
        return np.sum(self.elp)

