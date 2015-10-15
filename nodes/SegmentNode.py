"""
Node definining a segmentation of time into a sequence of Bernoulli posteriors.
"""
from __future__ import division
import numpy as np

from ..pelt import find_changepoints, calc_state_probs
from scipy.stats import bernoulli
from scipy.special import logit
import multiprocessing as mp


def find_cp(args):
    """
    Find changepoints for a given range of data in psi and write out
    state probabilities to xi. Called by SegmentNode.update(). Needs
    to be in module scope for pickling by multiprocessing.
    """
    start, stop, theta, alpha = args
    cplist = find_changepoints(psi[start:stop], theta, alpha)
    xi[start:stop, 1] = calc_state_probs(psi[start:stop], theta, cplist)
    xi[start:stop, 0] = 1 - xi[start:stop, 1]
    return cplist

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
    def __init__(self, z, chunklist, name='segmodel'):
        """
        z: mean value of the state in each time bin
        chunklist: iterable of [start, end) tuples for each subset that
            can be parallelized over
        """
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
        self.chunklist = chunklist
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
        # dimension of log_evidence = (T, M)

        # create global variables psi, x
        global psi, xi

        # make shared memory buffer, cast as numpy array
        psi = np.frombuffer(mp.Array('d', log_evidence.ravel(), lock=False))
        psi = psi.reshape(log_evidence.shape)
        # psi = log_evidence #- np.amax(log_evidence, axis=1, keepdims=True)

        xi = np.frombuffer(mp.Array('d', psi.size, lock=False))
        xi = xi.reshape(psi.shape)
        # xi = np.zeros_like(psi)

        # get needed parameters
        theta = self.nodes['z'].theta
        alpha = self.nodes['z'].alpha

        # pack everything needed for worker function into tuples
        arglist = [t + (theta, alpha) for t in self.chunklist]

        # initialize pool
        pool = mp.Pool()

        # map and reduce to get list of all changepoints
        allcp = pool.map(find_cp, arglist)
        cplist = sorted([cp for l in allcp for cp in l])
        pool.close()

        # get run starts
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
        elp = pi * logit(theta) - m * (alpha - np.log1p(-theta))
        self.elp[idx] = elp

        if self.update_finalizer is not None:
            self.update_finalizer(idx)

        return self

    def entropy(self):
        return np.sum(self.Hz)

    def expected_log_prior(self):
        return np.sum(self.elp)

