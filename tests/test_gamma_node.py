"""
Tests for Gamma node.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace 
import numpy as np
import numpy.testing as npt
from spiketopics.nodes.GammaNode import GammaNode
from scipy.special import digamma

priors_shape = np.array([1, 1, 2, 3]).reshape(2, 2)
priors_rate = np.array([0.1, 1, 0.1, 0.5]).reshape(2, 2)

post_shape = np.array([4, 1, 1, 1.1]).reshape(2, 2)
post_rate = np.array([0.1, 1, 2.1, 1.1]).reshape(2, 2)

prior_mat = np.dstack((priors_shape, priors_rate))
post_mat = np.dstack((post_shape, post_rate))

def test_init_sets_priors_and_posts():
    gn = GammaNode(prior=prior_mat, post=post_mat)
    npt.assert_array_equal(gn.prior_shape, priors_shape)
    npt.assert_array_equal(gn.prior_rate, priors_rate)
    npt.assert_array_equal(gn.post_shape, post_shape)
    npt.assert_array_equal(gn.post_rate, post_rate)

def test_misshapen_inputs_raises_error():
    pr_mat = prior_mat[..., 0][..., np.newaxis]
    po_mat = post_mat[..., 1][..., np.newaxis]
    assert_raises(ValueError, GammaNode, prior=pr_mat, post=post_mat)
    assert_raises(ValueError, GammaNode, prior=prior_mat, post=po_mat)

def test_can_set_name():
    gn = GammaNode(prior=prior_mat, post=post_mat, name='tester')
    assert_equals(gn.name, 'tester')

def test_expected_x():
    gn = GammaNode(prior=prior_mat, post=post_mat)
    npt.assert_array_equal(gn.expected_x(), post_shape / post_rate)

def test_expected_log_x():
    gn = GammaNode(prior=prior_mat, post=post_mat)
    log_x = digamma(post_shape) - np.log(post_rate)
    npt.assert_array_equal(gn.expected_log_x(), log_x)

def test_log_prior():
    gn = GammaNode(prior=prior_mat, post=post_mat)
    gn.expected_log_prior()

def test_entropy():
    gn = GammaNode(prior=prior_mat, post=post_mat)
    gn.entropy()
