"""
Tests for Markov transition node.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace 
import numpy as np
import numpy.testing as npt
from spiketopics.nodes.TransitionMatrixNode import TransitionMatrixNode
from scipy.special import digamma

prior_mat = np.array([0.1, 1, 3, 0.5]).reshape(2, 2)
post_mat = np.array([10, 2, 1., 5]).reshape(2, 2)

def test_init_sets_priors_and_posts():
    tn = TransitionMatrixNode(prior=prior_mat, post=post_mat)
    assert_equals(tn.M, prior_mat.shape[0])
    npt.assert_array_equal(tn.prior, prior_mat)
    npt.assert_array_equal(tn.post, post_mat)

def test_misshapen_inputs_raises_error():
    pr_mat = prior_mat[..., 0][..., np.newaxis]
    po_mat = post_mat[..., 1][..., np.newaxis]
    pm = np.array([10, 2, 1., 5, 6, 7, 2, 2, 2]).reshape(3, 3)
    assert_raises(ValueError, TransitionMatrixNode, prior=pr_mat, post=post_mat)
    assert_raises(ValueError, TransitionMatrixNode, prior=prior_mat, post=po_mat)
    assert_raises(ValueError, TransitionMatrixNode, prior=prior_mat, post=pm)

def test_can_set_name():
    tn = TransitionMatrixNode(prior=prior_mat, post=post_mat, name='tester')
    assert_equals(tn.name, 'tester')

def test_expected_x():
    tn = TransitionMatrixNode(prior=prior_mat, post=post_mat)
    alpha_sum = np.sum(tn.post, axis=0, keepdims=True)
    npt.assert_array_equal(tn.expected_x(), post_mat / alpha_sum)
    assert_equals(tn.expected_x().shape, post_mat.shape)

def test_expected_log_x():
    tn = TransitionMatrixNode(prior=prior_mat, post=post_mat)
    log_x = digamma(post_mat) / digamma(np.sum(post_mat, axis=0, keepdims=True))
    npt.assert_array_equal(tn.expected_log_x(), log_x)

def test_log_prior_shape():
    tn = TransitionMatrixNode(prior=prior_mat, post=post_mat)
    tn.expected_log_prior()

def test_entropy():
    tn = TransitionMatrixNode(prior=prior_mat, post=post_mat)
    tn.entropy()

def test_update_raises_notimplementederror():
    tn = TransitionMatrixNode(prior=prior_mat, post=post_mat)
    assert_raises(NotImplementedError, tn.update)