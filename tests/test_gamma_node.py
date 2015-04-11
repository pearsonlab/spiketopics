"""
Tests for Gamma node.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace 
import numpy as np
import numpy.testing as npt
from spiketopics.nodes.GammaNode import GammaNode
from scipy.special import digamma

prior_shape = np.array([1, 1, 2, 3]).reshape(2, 2)
prior_rate = np.array([0.1, 1, 0.1, 0.5]).reshape(2, 2)

post_shape = np.array([4, 1, 1, 1.1]).reshape(2, 2)
post_rate = np.array([0.1, 1, 2.1, 1.1]).reshape(2, 2)

gn = GammaNode(prior_shape=prior_shape, prior_rate=prior_rate, 
    post_shape=post_shape, post_rate=post_rate)

def test_init_sets_priors_and_posts():
    npt.assert_array_equal(gn.prior_shape, prior_shape)
    npt.assert_array_equal(gn.prior_rate, prior_rate)
    npt.assert_array_equal(gn.post_shape, post_shape)
    npt.assert_array_equal(gn.post_rate, post_rate)

def test_misshapen_inputs_raises_error():
    assert_raises(ValueError, GammaNode, prior_shape=prior_shape, 
        prior_rate=np.array(1.), post_shape=post_shape, post_rate=post_rate)
    assert_raises(ValueError, GammaNode, prior_shape=prior_shape, 
        prior_rate=prior_rate, post_shape=post_shape, 
        post_rate=np.array(1.))

def test_can_set_name():
    gn = GammaNode(prior_shape=prior_shape, prior_rate=prior_rate, 
        post_shape=post_shape, post_rate=post_rate, name='tester')
    assert_equals(gn.name, 'tester')

def test_expected_x():
    npt.assert_array_equal(gn.expected_x(), post_shape / post_rate)

def test_expected_log_x():
    log_x = digamma(post_shape) - np.log(post_rate)
    npt.assert_array_equal(gn.expected_log_x(), log_x)

def test_log_prior():
    gn.expected_log_prior()

def test_entropy():
    gn.entropy()
