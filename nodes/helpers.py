"""
Helper functions for dealing with nodes.
"""
from __future__ import division
import numpy as np
import scipy.stats as stats
from .GammaNode import GammaNode
from .GaussianNode import GaussianNode
from .DirichletNode import DirichletNode
from .NormalGammaNode import NormalGammaNode
from .HMM import MarkovChainNode, HMMNode, DurationNode
from .utility_nodes import ProductNode

def check_shapes(par_shapes, pars):
    """
    Check for consistency between par_shapes, a dict with (name, shape)
    pairs, and pars, a dict of (name, value) pairs. 
    """
    for var, shape in par_shapes.iteritems():
        if var not in pars:
            raise ValueError('Argument missing: {}'.format(var))
        elif pars[var].shape != shape:
            raise ValueError('Argument has wrong shape: {}'.format(var))

def initialize_gamma(name, node_shape, **kwargs):
    """
    Initialize a gamma variable.
    """
    pars = {k: np.array(v) for k, v in kwargs.iteritems()}
    par_shapes = ({'prior_shape': node_shape, 'prior_rate': node_shape,
        'post_shape': node_shape, 'post_rate': node_shape })

    check_shapes(par_shapes, pars)

    node = GammaNode(pars['prior_shape'], pars['prior_rate'], 
        pars['post_shape'], pars['post_rate'], name=name)

    node.has_parents = False

    return (node,)

def initialize_gamma_hierarchy(basename, parent_shape, 
    child_shape, **kwargs):
    """
    Initialize a hierarchical gamma variable: 
        lambda ~ Gamma(c, c * m)
        c ~ Gamma
        m ~ Gamma

    basename is the name of the name of the child variable 
    parent_shape and child_shape are the shapes of the parent and
        child variables
    """
    pars = {k: np.array(v) for k, v in kwargs.iteritems()}
    par_shapes = ({'prior_shape_shape': parent_shape, 
        'prior_shape_rate': parent_shape, 
        'prior_mean_shape': parent_shape, 'prior_mean_rate': parent_shape, 
        'post_shape_shape': parent_shape, 'post_shape_rate': parent_shape, 
        'post_mean_shape': parent_shape, 'post_mean_rate': parent_shape, 
        'post_child_shape': child_shape, 'post_child_rate': child_shape})

    check_shapes(par_shapes, pars)
    
    shapename = basename + '_shape'
    shape = GammaNode(pars['prior_shape_shape'], 
        pars['prior_shape_rate'], pars['post_shape_shape'], 
        pars['post_shape_rate'], name=shapename)

    meanname = basename + '_mean'
    mean = GammaNode(pars['prior_mean_shape'], 
        pars['prior_mean_rate'], pars['post_mean_shape'], 
        pars['post_mean_rate'], name=meanname)

    child = GammaNode(shape, ProductNode(shape, mean),
        pars['post_child_shape'], pars['post_child_rate'], 
        name=basename)

    def update_shape(idx=Ellipsis):
        """
        Update for shape variable in terms of mean and child nodes.
        idx is assumed to index *last* dimension of arrays
        """ 
        # calculate number of units
        n_units = np.sum(np.ones(child_shape).T[idx])

        shape.post_shape.T[idx] = shape.prior_shape.T[idx] 
        shape.post_shape.T[idx] += 0.5 * n_units

        shape.post_rate.T[idx] = shape.prior_rate.T[idx] 
        eff_rate = (mean.expected_x() * child.expected_x() - 
            mean.expected_log_x() - child.expected_log_x() - 1)
        shape.post_rate.T[idx] += np.sum(eff_rate.T[idx])

    def update_mean(idx=Ellipsis):
        """
        Update for mean variable in terms of shape and child nodes.
        idx is assumed to index *last* dimension of arrays
        """
        # calculate number of units
        n_units = np.sum(np.ones(child_shape).T[idx])

        mean.post_shape.T[idx] = mean.prior_shape.T[idx]
        mean.post_shape.T[idx] += n_units * shape.expected_x().T[idx]
       
        mean.post_rate.T[idx] = mean.prior_rate.T[idx]
        mean.post_rate.T[idx] += (shape.expected_x().T[idx] * 
            np.sum(child.expected_x().T[idx]))

    def update_parents(idx=Ellipsis):
        shape.update(idx)
        mean.update(idx)

    shape.update = update_shape
    mean.update = update_mean
    child.has_parents = True
    child.update_parents = update_parents

    return (shape, mean, child)

def initialize_HMM(n_chains, n_states, n_times, n_durations=None, **kwargs):
    """
    Initialize nodes that compose a Hidden Markov Model. 
    Nodes are:
        z: latent states (taking values 0...n_states-1 x n_times times; 
            0...n_chains-1 copies)
        A: n_chains copies of a n_states x n_states transition matrix
        pi: n_chains copies of a n_states vector of initial probabilities

    Returns a tuple of three nodes (z, A, pi)
    """
    K = n_chains
    M = n_states
    T = n_times

    # if we specified durations, it's semi-Markov
    hsmm = n_durations is not None

    par_shapes = ({'A_prior': (M, M, K), 'A_post': (M, M, K),
        'pi_prior': (M, K), 'pi_post': (M, K), 'z_init': (M, T, K), 
        'zz_init': (M, M, T - 1, K), 'logZ_init': (K,)})

    if hsmm:
        d_pars = ({'prior_mean': kwargs['d_prior_mean'], 
            'prior_scaling': kwargs['d_prior_scaling'], 
            'prior_shape': kwargs['d_prior_shape'], 
            'prior_rate': kwargs['d_prior_rate'], 
            'post_mean': kwargs['d_post_mean'], 
            'post_scaling': kwargs['d_post_scaling'], 
            'post_shape': kwargs['d_post_shape'], 
            'post_rate': kwargs['d_post_rate']})

    check_shapes(par_shapes, kwargs)

    A = DirichletNode(kwargs['A_prior'], kwargs['A_post'], name='A')
    pi = DirichletNode(kwargs['pi_prior'], kwargs['pi_post'], name='pi')
    z = MarkovChainNode(kwargs['z_init'], kwargs['zz_init'], 
        kwargs['logZ_init'], name='z')
    if hsmm:
        d = initialize_lognormal_duration_node(n_chains, n_states, 
            n_durations, **d_pars)

    if hsmm:
        node = HMMNode(z, A, pi, d)
    else:
        node = HMMNode(z, A, pi)

    return (node,)

def initialize_gaussian(name, node_shape, prior_shape, **kwargs):
    """
    Initialize a gaussian variable.
    """
    pars = {k: np.array(v) for k, v in kwargs.iteritems()}
    par_shapes = ({'prior_mean': prior_shape, 'prior_prec': prior_shape,
        'post_mean': node_shape, 'post_prec': node_shape })

    check_shapes(par_shapes, pars)

    node = GaussianNode(pars['prior_mean'], pars['prior_prec'], 
        pars['post_mean'], pars['post_prec'], name=name)

    node.has_parents = False

    return (node,)

def initialize_gaussian_hierarchy(basename, child_shape, parent_shape, 
    grandparent_shape, **kwargs):
    """
    Initialize a hierarchical gaussian variable: 
        x ~ N(0, v^2)
        v^2 ~ Inv-Gamma(s, r)

    basename is the name of the name of the child variable 
    child_shape is the shape of x
    parent_shape is the shape of v^2
    grandparent_shape is the shape of s and r

    Our strategy is to put a Gamma node prior on tau = 1/v^2 
    """
    pars = {k: np.array(v) for k, v in kwargs.iteritems()}
    par_shapes = ({
        'prior_prec_shape': grandparent_shape, 
        'prior_prec_rate': grandparent_shape, 
        'post_mean': child_shape,
        'post_prec': child_shape,
        'post_prec_shape': parent_shape,
        'post_prec_rate': parent_shape
        })

    check_shapes(par_shapes, pars)

    prec_name = basename + '_prec'
    prec = GammaNode(pars['prior_prec_shape'], 
        pars['prior_prec_rate'], pars['post_prec_shape'], 
        pars['post_prec_rate'], name=prec_name)

    prior_mean = np.zeros(parent_shape)
    child = GaussianNode(prior_mean, prec,
        pars['post_mean'], pars['post_prec'], name=basename)

    def update_prec(idx=Ellipsis):
        """
        Update for precision variable in terms of child node.
        idx is assumed to index *last* dimension of arrays
        """ 
        # calculate number of units
        n_units = np.sum(np.ones(child_shape).T[idx])

        arr_list = np.broadcast_arrays(prec.post_shape.T, 
            prec.prior_shape.T)
        prec.post_shape.T[idx] = arr_list[1][idx] 
        prec.post_shape.T[idx] += 0.5 * n_units

        arr_list = np.broadcast_arrays(prec.post_rate.T, 
            prec.prior_rate.T)
        prec.post_rate.T[idx] = arr_list[1][idx]
        eff_rate = 0.5 * (child.expected_var_x() + child.expected_x() ** 2)
        prec.post_rate.T[idx] += np.sum(eff_rate.T[idx])

    def update_parents(idx=Ellipsis):
        prec.update(idx)

    prec.update = update_prec
    child.update_parents = update_parents
    child.has_parents = True

    return (prec, child)

def initialize_lognormal_duration_node(n_chains, n_states, n_durations, 
    **kwargs):
    """
    Initialize a lognormal distribution with normal-gamma priors to be 
    used in a hidden semi-markov model.
    """
    K = n_chains
    M = n_states
    D = n_durations

    par_shapes = ({'prior_mean': (M, K), 'prior_scaling': (M, K), 
        'prior_shape': (M, K), 'prior_rate': (M, K), 
        'post_mean': (M, K), 'post_scaling': (M, K), 
        'post_shape': (M, K), 'post_rate': (M, K)})

    check_shapes(par_shapes, kwargs)

    parent_node = NormalGammaNode(kwargs['prior_mean'], 
        kwargs['prior_scaling'], kwargs['prior_shape'], 
        kwargs['prior_rate'], kwargs['post_mean'], 
        kwargs['post_scaling'], kwargs['post_shape'],
        kwargs['post_rate'])

    # make durations (must be > 0):
    # dvec should have one vector for each chain (could be different)
    dvec = np.tile(np.arange(1, D + 1), (K, 1)).T

    # make node
    node = DurationNode(M, dvec, parent_node)

    def logpd(self):
        """
        Calculate E[log p(d|z)] under the variational posterior.
        logpd should be M x D x K
        """
        parent = self.parent
        t = parent.expected_t()
        logt = parent.expected_log_t()
        tx = parent.expected_tx()
        txx = parent.expected_txx()
        dv = self.dvec[:, np.newaxis, :]

        logpd = (0.5 * np.log(2 * np.pi) - 
            0.5 * logt[np.newaxis, ...] -
            np.log(dv) - 0.5 * txx[np.newaxis, ...] + 
            np.log(dv) * tx - 
            0.5 * (np.log(dv) ** 2) * t)

        # normalize
        logpd /= np.sum(logpd, axis=0, keepdims=True)

        return logpd.transpose((1, 0, 2))

    def calc_ess(self, idx):
        """
        Calculate expected sufficient statistics to pass on to parent node 
        (in this case, log normal).
        """
        C = self.C[..., idx]
        dvec = self.dvec[..., idx]

        # calculate some summaries (axis 1 = d axis)
        Csum = np.sum(C, axis=1)
        Clogd = np.sum(C * np.log(dvec), axis=1)
        Clogd2 = np.sum(C * np.log(dvec)**2, axis=1)

        # calculate sufficient stats for natural parameters
        eta1 = 0.5 * Csum
        eta2 = -0.5 * Clogd2
        eta3 = Clogd
        eta4 = Csum

        return (eta1, eta2, eta3, eta4)

    # bind these methods to the duration node
    node.logpd = logpd.__get__(node, DurationNode)
    node.calc_ess = calc_ess.__get__(node, DurationNode)

    return node
