"""
Tests for Gaussian node.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace 
import numpy as np
import numpy.testing as npt
from spiketopics.nodes.GaussianNode import GaussianNode

prior_mean = np.array([1, 1, 2, 3]).reshape(2, 2)
prior_prec = np.array([0.1, 1, 0.1, 0.5]).reshape(2, 2)

post_mean = np.array([4, 1, 1, 1.1]).reshape(2, 2)
post_prec = np.array([0.1, 1, 2.1, 1.1]).reshape(2, 2)

gn = GaussianNode(prior_mean=prior_mean, prior_prec=prior_prec, 
    post_mean=post_mean, post_prec=post_prec)

def test_init_sets_priors_and_posts():
    npt.assert_array_equal(gn.prior_mean, prior_mean)
    npt.assert_array_equal(gn.prior_prec, prior_prec)
    npt.assert_array_equal(gn.post_mean, post_mean)
    npt.assert_array_equal(gn.post_prec, post_prec)

def test_misshapen_inputs_raises_error():
    assert_raises(ValueError, GaussianNode, prior_mean=prior_mean, 
        prior_prec=np.array(1.), post_mean=post_mean, post_prec=post_prec)
    assert_raises(ValueError, GaussianNode, prior_mean=prior_mean, 
        prior_prec=prior_prec, post_mean=post_mean, 
        post_prec=np.array(1.))

def test_can_set_name():
    gn = GaussianNode(prior_mean=prior_mean, prior_prec=prior_prec, 
        post_mean=post_mean, post_prec=post_prec, name='tester')
    assert_equals(gn.name, 'tester')

def test_expected_x():
    npt.assert_array_equal(gn.expected_x(), post_mean)

def test_expected_var_x():
    npt.assert_array_equal(gn.expected_var_x(), 1. / post_prec)

def test_expected_prec_x():
    npt.assert_array_equal(gn.expected_prec_x(), post_prec)

def test_expected_exp_x():
    npt.assert_array_equal(gn.expected_exp_x(), 
        np.exp(post_mean + 0.5 / post_prec))

def test_log_prior():
    gn.expected_log_prior()

def test_entropy():
    gn.entropy()

