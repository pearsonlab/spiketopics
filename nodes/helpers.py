"""
Helper functions for dealing with nodes.
"""
from __future__ import division
from .GammaNode import GammaNode
from .DirichletNode import DirichletNode
from .HMM import MarkovChainNode, HMMNode
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
    par_shapes = ({'prior_shape': node_shape, 'prior_rate': node_shape,
        'post_shape': node_shape, 'post_rate': node_shape })

    check_shapes(par_shapes, kwargs)

    node = GammaNode(kwargs['prior_shape'], kwargs['prior_rate'], 
        kwargs['post_shape'], kwargs['post_rate'], name=name)

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
    par_shapes = ({'prior_shape_shape': parent_shape, 
        'prior_shape_rate': parent_shape, 
        'prior_mean_shape': parent_shape, 'prior_mean_rate': parent_shape, 
        'post_shape_shape': parent_shape, 'post_shape_rate': parent_shape, 
        'post_mean_shape': parent_shape, 'post_mean_rate': parent_shape, 
        'post_child_shape': child_shape, 'post_child_rate': child_shape})

    check_shapes(par_shapes, kwargs)
    
    shapename = basename + '_shape'
    shape = GammaNode(kwargs['prior_shape_shape'], 
        kwargs['prior_shape_rate'], kwargs['post_shape_shape'], 
        kwargs['post_shape_rate'], name=shapename)

    meanname = basename + '_mean'
    mean = GammaNode(kwargs['prior_mean_shape'], 
        kwargs['prior_mean_rate'], kwargs['post_mean_shape'], 
        kwargs['post_mean_rate'], name=meanname)

    child = GammaNode(shape, ProductNode(shape, mean),
        kwargs['post_child_shape'], kwargs['post_child_rate'], 
        name=basename)

    # to do: wire up update functions for mean and shape here, since
    # all the necessary nodes are present
    # might even wire up update for child to autocall these before calling
    # a personal update method (to be set in model)

    return (shape, mean, child)

def initialize_HMM(n_chains, n_states, n_times, **kwargs):
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
    par_shapes = ({'A_prior': (M, M, K), 'A_post': (M, M, K),
        'pi_prior': (M, K), 'pi_post': (M, K), 'z_prior': (M, T, K), 
        'zz_prior': (M, M, T - 1, K), 'logZ_prior': (K,)})

    check_shapes(par_shapes, kwargs)

    A = DirichletNode(kwargs['A_prior'], kwargs['A_post'], name='A')
    pi = DirichletNode(kwargs['pi_prior'], kwargs['pi_post'], name='pi')
    z = MarkovChainNode(kwargs['z_prior'], kwargs['zz_prior'], 
        kwargs['logZ_prior'], name='z')

    return HMMNode(z, A, pi)
