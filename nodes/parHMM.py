"""
Nodes related to Hidden Markov Models
"""
from __future__ import division
import numpy as np
try:
    from ..forward_backward import fb_infer
except ImportError:
    from ..fbi import fb_infer
import multiprocessing as mp

def do_fb(args):
    """
    Run forward-backward on each independent stimulus.
    """
    idx, start, stop, A, pi = args
    slc = slice(start, stop + 1)
    le = np.exp(psi[slc, :])
    xi_t, logZ_t, Xi_t = fb_infer(A, pi, le)
    xi[slc] = xi_t
    Xi[start] = 0.  # boundary between stims: shouldn't contribute to ELBO
    Xi[(start + 1):(stop + 1)] = Xi_t
    logZ_vec[idx] = logZ_t


class MarkovChainNode:
    """
    Distribution for a set of Markov chains, the variables in each taking on
    discrete states defined by a Dirichlet distribution.
    Variables are:
        z: Markov chain (states x times x copies)
        zz: two-slice marginals for transitions (states x states x
            times - 1, copies)
        logZ: log partition function for posterior (copies)

    Note that this is not a normal node, since it does not parameterize the
        full joint distribution over z. Rather, it is only a container for
        the expected values of the marginals z and zz, and the log
        partition function, logZ.
    """
    def __init__(self, z_init, zz_init, logZ_init, name='markov'):
        if len(z_init.shape) < 2:
            raise ValueError(
                'State variable must have at least two dimensions')
        if len(zz_init.shape) < 3:
            raise ValueError(
                'Transition marginal must have at least three dimensions')

        self.M = z_init.shape[0]
        self.T = z_init.shape[1]
        if len(z_init.shape) > 2:
            self.K = z_init.shape[2:]
        else:
            self.K = (1,)

        if zz_init.shape[0:3] != (self.M, self.M, self.T - 1):
            raise ValueError(
                'Transition marginal has shape inconsistent with prior')
        if logZ_init.shape != self.K:
            raise ValueError(
                'logZ has shape inconsistent with prior')

        self.name = name
        self.shape = z_init.shape

        # set inits (but normalize)
        self.z = z_init / np.sum(z_init, axis=0, keepdims=True)
        self.zz = zz_init / np.sum(zz_init, axis=(0, 1), keepdims=True)
        self.logZ = logZ_init.copy()

    def update(self, idx, new_z, new_zz, new_logZ):
        self.z[:, :, idx] = new_z
        self.zz[:, :, :, idx] = new_zz
        self.logZ[idx] = new_logZ

        return self

class DurationNode:
    """
    Node encapsulating a duration distribution for states in a semi-Markov
    model. Requires a vector of duration values and a parent node specifying
    parameters for the distribution.
    """
    def __init__(self, num_states, duration_vals, parent_node,
        name='duration'):
        D = duration_vals.shape[0]
        K = duration_vals.shape[1:]
        M = num_states

        self.D = D
        self.dvec = duration_vals.copy()
        self.parent = parent_node
        self.C = np.zeros((M, D) + K)

    def logpd(self):
        """
        Return log probability of all duration values in dvec. Array should
        be M x D x ..., where M is the number of levels of each hidden state
        and D the total number of durations considered.
        """
        raise NotImplementedError('Instances must supply this method.')

    def get_durations(self):
        """
        Return an array (D x ...) of possible hidden state durations.
        M is the number of levels of each hidden state, D the number of
        durations, and other indices denote replicates.
        """
        return self.dvec

    def update(self, idx, C):
        """
        Given sufficient statistics C from the forward-backward algorithm,
        convert to expected sufficient statistics suitable for the parent
        node and update the distribution.
        """
        self.C[..., idx] = C
        ess = self.calc_ess(idx)
        self.parent.update(idx, *ess)

        return self

    def calc_ess(self, idx):
        """
        Calculate expected sufficient statistics in a form suitable for
        updating parent node. Return a dict of named arguments for
        passing to parent node's update method.
        """
        raise NotImplementedError('Instances must supply this method.')

    def entropy(self):
        """
        Return differential entropy for this piece. This is just the entropy
        of the parameter distribution.
        """
        return self.parent.entropy()

    def expected_log_prior(self):
        """
        Expected value of log prior under the posterior.
        """
        return self.parent.expected_log_prior()

    def expected_log_duration_prob(self):
        """
        Expected value of the piece of log state sequence depending upon
        p(d|z).
        """
        C = self.C
        return np.sum(C * self.logpd())

class parHMMNode:
    """
    Node representing a Hidden Markov Model. Comprises a MarkovChainNode for
    the latent states, a DirichletNode for the transition matrix, and a
    DirichletNode for the initial state probabilities.
    If a d (=duration) node of type DurationNode is supplied, the model is a
    Hidden Semi-Markov Model (HSMM).
    """
    def __init__(self, z, A, pi, chunklist, d=None, name='HMM'):
        M = z.shape[0]
        T = z.shape[1]
        if len(z.shape) > 2:
            K = z.shape[2:]
        else:
            K = (1,)

        if pi.shape != (M,) + K :
            raise ValueError('Initial state shape inconsistent with latents')
        if A.shape != (M, M) + K :
            raise ValueError(
                'Transition matrix shape inconsistent with latents')

        self.hsmm = d is not None
        self.chunklist = chunklist

        self.M = M
        self.T = T
        self.K = K
        self.shape = z.shape
        self.name = name
        self.update_finalizer = None

        self.nodes = {'z': z, 'A': A, 'pi': pi}
        if self.hsmm:
            self.nodes.update({'d': d})

        self.Hz = np.zeros(K)

    def update(self, idx, log_evidence):
        """
        Update the HMM with index idx.
        idx needs to be an object capable of indexing the relevant Markov
            Chain (i.e., z after the first two (M, T) indices, A after the
            first two (M, M) indices, etc.)
        log_evidence is log p(y|z, theta) (T, M) where y is the
            observed data from the HMM, z is the latent state variable(s),
            and theta are the other parameters of the HMM
        """
        # create globals (needed for parallelization)
        global psi, xi, Xi, logZ_vec

        # calculate some sufficient statistics
        # this is okay for chunks because we set Xi = 0 for all
        # time points that are boundaries between stims
        zz = self.nodes['z'].zz[..., idx]
        zz_bar = np.sum(zz, axis=2)  # sum over time

        # sum up sufficient stats for all trial starts
        start_list = [s for s, _ in self.chunklist]
        z0 = np.sum(self.nodes['z'].z[:, start_list, idx], axis=1)

        ########### update nodes
        self.nodes['A'].update(idx, zz_bar)
        self.nodes['pi'].update(idx, z0)

        ########### update chains
        T, M = log_evidence.shape
        psi_r = log_evidence - np.amax(log_evidence, axis=1, keepdims=True)
        psi = np.frombuffer(mp.Array('d', psi_r.ravel(), lock=False))
        psi = psi.reshape((T, M))

        xi = np.frombuffer(mp.Array('d', T * M, lock=False))
        xi = xi.reshape((T, M))

        Xi = np.frombuffer(mp.Array('d', T * M * M, lock=False))
        Xi = Xi.reshape((T, M, M))

        logZ_vec = np.frombuffer(mp.Array('d', len(self.chunklist), lock=False))
        logZ_vec = logZ_vec.reshape((len(self.chunklist),))

        # calculate variational parameters in z posterior
        A_par = self.nodes['A'].expected_log_x()[..., idx]
        pi_par = self.nodes['pi'].expected_log_x()[..., idx]
        Ap = np.exp(A_par)
        pip = np.exp(pi_par)

        # pack everything needed for worker function into tuples
        arglist = [(i,) + t + (Ap, pip) for i, t in enumerate(self.chunklist)]

        # initialize pool
        pool = mp.Pool()

        # map and reduce to get list of all changepoints
        pool.map(do_fb, arglist)
        pool.close()

        # post-processing
        logZ = np.sum(logZ_vec)
        xi = xi.T  # now (M, T)
        # now cut off Xi(t = 0), which is 0 anyway
        Xi = Xi.transpose((1, 2, 0))[:, :, 1:]  # now (M, M, T - 1)
        self.nodes['z'].update(idx, xi, Xi, logZ)

        ########### calculate entropy pieces
        emission_piece = np.sum(xi.T * psi)
        initial_piece = np.sum(pi_par.dot(xi[:, start_list]))
        transition_piece = np.sum(Xi * A_par[..., np.newaxis])
        logq = emission_piece + initial_piece + transition_piece
        self.Hz[idx] = -logq + logZ

        if self.hsmm:
            duration_piece = np.sum(C * logpd)
            self.Hz[idx] += -duration_piece

        assert(np.all((self.Hz >= 0) | np.isclose(self.Hz, 0)))

        if self.update_finalizer is not None:
            self.update_finalizer(idx)

        return self

    def entropy(self):
        H = (np.sum(self.Hz) + self.nodes['A'].entropy() +
            self.nodes['pi'].entropy())
        if self.hsmm:
            H += self.nodes['d'].entropy()

        return H

    def expected_log_prior(self):
        elp = (self.nodes['A'].expected_log_prior() +
            self.nodes['pi'].expected_log_prior())
        if self.hsmm:
            elp += self.nodes['d'].expected_log_prior()

        return elp

    def expected_log_state_sequence(self):
        """
        Return expected log probability of the sequence z
        """
        # get starts of all stims
        start_list = [s for s, _ in self.chunklist]
        xi = self.nodes['z'].z
        Xi = self.nodes['z'].zz
        logpi = self.nodes['pi'].expected_log_x()
        logA = self.nodes['A'].expected_log_x()

        elss = (np.sum(xi[:, start_list, :] * logpi[:, np.newaxis, :]) + np.sum(Xi * logA[:, :, np.newaxis]))

        if self.hsmm:
            elss += self.nodes['d'].expected_log_duration_prob()

        return elss
