"""
Helper functions for dealing with nodes.
"""
from __future__ import division
import numpy as np
from .GammaNode import GammaNode
from .GaussianNode import GaussianNode
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
        'pi_prior': (M, K), 'pi_post': (M, K), 'z_init': (M, T, K), 
        'zz_init': (M, M, T - 1, K), 'logZ_init': (K,)})

    check_shapes(par_shapes, kwargs)

    A = DirichletNode(kwargs['A_prior'], kwargs['A_post'], name='A')
    pi = DirichletNode(kwargs['pi_prior'], kwargs['pi_post'], name='pi')
    z = MarkovChainNode(kwargs['z_init'], kwargs['zz_init'], 
        kwargs['logZ_init'], name='z')

    return (HMMNode(z, A, pi),)

def initialize_gaussian(name, node_shape, **kwargs):
    """
    Initialize a gaussian variable.
    """
    pars = {k: np.array(v) for k, v in kwargs.iteritems()}
    par_shapes = ({'prior_mean': node_shape, 'prior_prec': node_shape,
        'post_mean': node_shape, 'post_prec': node_shape })

    check_shapes(par_shapes, pars)

    node = GaussianNode(pars['prior_mean'], pars['prior_prec'], 
        pars['post_mean'], pars['post_prec'], name=name)

    node.has_parents = False

    return (node,)

