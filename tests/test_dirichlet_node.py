"""
Tests for Dirichlet node.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace 
import numpy as np
import numpy.testing as npt
from spiketopics.nodes.DirichletNode import DirichletNode
from scipy.special import digamma

prior_mat = np.array([0.1, 1, 3, 0.5]).reshape(2, 2)
post_mat = np.array([10, 2, 1., 5]).reshape(2, 2)

def test_init_sets_priors_and_posts():
    dn = DirichletNode(prior=prior_mat, post=post_mat)
    assert_equals(dn.M, prior_mat.shape[0])
    npt.assert_array_equal(dn.prior, prior_mat)
    npt.assert_array_equal(dn.post, post_mat)

def test_misshapen_inputs_raises_error():
    pm = np.array([10, 2, 1., 5, 6, 7, 2, 2, 2]).reshape(3, 3)
    assert_raises(ValueError, DirichletNode, prior=prior_mat, post=pm)

def test_can_set_name():
    dn = DirichletNode(prior=prior_mat, post=post_mat, name='tester')
    assert_equals(dn.name, 'tester')

def test_expected_x():
    dn = DirichletNode(prior=prior_mat, post=post_mat)
    alpha_sum = np.sum(dn.post, axis=0, keepdims=True)
    npt.assert_array_equal(dn.expected_x(), post_mat / alpha_sum)
    assert_equals(dn.expected_x().shape, post_mat.shape)

def test_expected_log_x():
    dn = DirichletNode(prior=prior_mat, post=post_mat)
    log_x = digamma(post_mat) / digamma(np.sum(post_mat, axis=0, keepdims=True))
    npt.assert_array_equal(dn.expected_log_x(), log_x)

def test_log_prior_shape():
    dn = DirichletNode(prior=prior_mat, post=post_mat)
    dn.expected_log_prior()

def test_entropy():
    dn = DirichletNode(prior=prior_mat, post=post_mat)
    dn.entropy()

def test_update_raises_notimplementederror():
    dn = DirichletNode(prior=prior_mat, post=post_mat)
    assert_raises(NotImplementedError, dn.update)