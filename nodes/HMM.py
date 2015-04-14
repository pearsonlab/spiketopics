"""
Nodes related to Hidden Markov Models
"""
import numpy as np
from ..forward_backward import fb_infer

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
    def __init__(self, z_prior, zz_prior, logZ_prior, name='markov'):
        if len(z_prior.shape) < 2:
            raise ValueError(
                'State variable must have at least two dimensions')
        if len(zz_prior.shape) < 3:
            raise ValueError(
                'Transition marginal must have at least three dimensions')

        self.M = z_prior.shape[0]
        self.T = z_prior.shape[1]
        if len(z_prior.shape) > 2:
            self.K = z_prior.shape[2:]
        else:
            self.K = (1,)

        if zz_prior.shape[0:3] != (self.M, self.M, self.T - 1):
            raise ValueError(
                'Transition marginal has shape inconsistent with prior')
        if logZ_prior.shape != self.K:
            raise ValueError(
                'logZ has shape inconsistent with prior')

        self.name = name
        self.shape = z_prior.shape
        self.z = z_prior
        self.zz = zz_prior
        self.logZ = logZ_prior

    def update(self, idx, new_z, new_zz, new_logZ):
        self.z[:, :, idx] = new_z
        self.zz[:, :, :, idx] = new_zz
        self.logZ[idx] = new_logZ

        return self

class HMMNode:
    """
    Node representing a Hidden Markov Model. Comprises a MarkovChainNode for 
    the latent states, a DirichletNode for the transition matrix, and a 
    DirichletNode for the initial state probabilities.
    """
    def __init__(self, z, A, pi):
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

        self.M = M
        self.T = T
        self.K = K
        self.shape = z.shape

        self.nodes = {'z': z, 'A': A, 'pi': pi} 
        self.Hz = np.empty(K)

    def update(self, idx, log_evidence):
        """
        Update the HMM with index idx.
        idx needs to be an object capable of indexing the relevant Markov
            Chain (i.e., z after the first two (M, T) indices, A after the
            first two (M, M) indices, etc.)
        log_evidence is log p(y|z, theta) (y.shape, T) where y is the 
            observed data from the HMM, z is the latent state variable(s), 
            and theta are the other parameters of the HMM
        """

        # calculate some sufficient statistics
        zz = self.nodes['z'].zz[:, :, idx]
        zz_bar = np.sum(zz, axis=2)  # sum over time
        z0 = self.nodes['z'].z[:, 0, idx]  # time 0 only

        ########### update nodes
        self.nodes['A'].update(zz_bar)
        self.nodes['pi'].update(z0)

        ########### update chains
        psi = log_evidence

        # calculate variational parameters in z posterior 
        A_par = self.nodes['A'].expected_log_x()
        pi_par = self.nodes['pi'].expected_log_x()

        xi, logZ, Xi = fb_infer(np.exp(A_par), np.exp(pi_par), np.exp(psi))
        self.nodes['z'].update(idx, xi, Xi, logZ) 

        ########### calculate entropy pieces
        emission_piece = xi.dot(psi)
        initial_piece = xi[:, 0].dot(pi_par)
        transition_piece = np.sum(Xi * A_par)
        logq = emission_piece + initial_piece + transition_piece
        self.Hz[idx] = -logq + logZ

        return self

    def entropy(self):
        return (np.sum(self.Hz) + self.nodes['A'].entropy() + 
            self.nodes['pi'].entropy())

    def expected_log_prior(self):
        return (self.nodes['A'].expected_log_prior() + 
            self.nodes['pi'].expected_log_prior())