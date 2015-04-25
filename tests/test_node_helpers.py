"""
Tests for node helper functions model.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace
import numpy as np
import numpy.testing as npt
import spiketopics.nodes as nd

def test_check_shapes_raises_error():
    par_shapes = {'a': (10, 10), 'b': (5,)}

    # first, check valid inputs raise no errors
    pars = {'a': np.ones((10, 10)), 'b': np.ones((5,))}
    nd.check_shapes(par_shapes, pars)

    # now, check missing input raises error 
    pars = {'a': np.ones((10, 10))}
    assert_raises(ValueError, nd.check_shapes, par_shapes, pars)

    # now, check misshapen input raises error 
    pars = {'a': np.ones((10, 1)), 'b': np.ones((5,))}
    assert_raises(ValueError, nd.check_shapes, par_shapes, pars)

def test_can_initialize_gamma():
    child_shape = (10, 5)
    basename = 'foo'
    cs = np.ones(child_shape)
    vals = ({'prior_shape': cs, 'prior_rate': cs, 
        'post_shape': cs, 'post_rate': cs })

    nodes = nd.initialize_gamma(basename, child_shape, **vals)

    assert_equals(len(nodes), 1)
    assert_equals({'foo'}, {n.name for n in nodes})
    assert_is_instance(nodes[0], nd.GammaNode)

def test_can_initialize_gamma_hierarchy():
    parent_shape = (5,)
    child_shape = (10, 5)
    basename = 'foo'
    ps = np.ones(parent_shape)
    cs = np.ones(child_shape)
    vals = ({'prior_shape_shape': ps, 'prior_shape_rate': ps, 
        'prior_mean_shape': ps, 'prior_mean_rate': ps,
        'post_shape_shape': ps, 'post_shape_rate': ps,
        'post_mean_shape': ps, 'post_mean_rate': ps,
        'post_child_shape': cs, 'post_child_rate': cs})

    nodes = nd.initialize_gamma_hierarchy(basename, parent_shape, 
        child_shape, **vals)

    assert_equals(len(nodes), 3)
    assert_equals({'foo', 'foo_shape', 'foo_mean'}, 
        {n.name for n in nodes})
    assert_is_instance(nodes[0], nd.GammaNode)

def test_hierarchy_updates():
    parent_shape = (5, 6)
    child_shape = (10, 5, 6)
    basename = 'foo'
    ps = np.ones(parent_shape)
    cs = np.ones(child_shape)
    vals = ({'prior_shape_shape': ps, 'prior_shape_rate': ps, 
        'prior_mean_shape': ps, 'prior_mean_rate': ps,
        'post_shape_shape': ps, 'post_shape_rate': ps,
        'post_mean_shape': ps, 'post_mean_rate': ps,
        'post_child_shape': cs, 'post_child_rate': cs})

    nodes = nd.initialize_gamma_hierarchy(basename, parent_shape, 
        child_shape, **vals)
    node_dict = {n.name: n for n in nodes}

    shape = node_dict['foo_shape']
    mean = node_dict['foo_mean']
    shape.update(1)
    mean.update(2)
    shape.update((0, 1))

def test_can_initialize_HMM():
    K = 5
    M = 3
    T = 50
    A_shape = (M, M, K)
    pi_shape = (M, K)
    z_shape = (M, T, K)
    zz_shape = (M, M, T - 1, K)
    logZ_shape = (K,)
    A = np.ones(A_shape)
    pi = np.ones(pi_shape)
    z = np.ones(z_shape)
    zz = np.ones(zz_shape)
    logZ = np.ones(logZ_shape)
    vals = ({'A_prior': A, 'A_post': A, 'pi_prior': pi, 'pi_post': pi, 
        'z_init': z, 'zz_init': zz, 'logZ_init': logZ})

    nodes = nd.initialize_HMM(K, M, T, **vals)
    hmm_node = nodes[0]
    node_dict = hmm_node.nodes

    assert_equals(len(nodes), 1)
    assert_equals({'A', 'pi', 'z'}, set(node_dict.keys()))
    assert_is_instance(node_dict['A'], nd.DirichletNode)
    assert_is_instance(node_dict['pi'], nd.DirichletNode)
    assert_is_instance(node_dict['z'], nd.MarkovChainNode)

def test_can_initialize_gaussian():
    child_shape = (10, 5)
    basename = 'foo'
    cs = np.ones(child_shape)
    vals = ({'prior_mean': cs, 'prior_prec': cs, 
        'post_mean': cs, 'post_prec': cs })

    nodes = nd.initialize_gaussian(basename, child_shape, **vals)

    assert_equals(len(nodes), 1)
    assert_equals({'foo'}, {n.name for n in nodes})
    assert_is_instance(nodes[0], nd.GaussianNode)

