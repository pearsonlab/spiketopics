"""
Tests for Hidden Markov model nodes.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace 
import numpy as np
import numpy.testing as npt
import spiketopics.nodes as nd

np.random.seed(12345)

K = 5
M = 3
T = 50
D = 35
A_shape = (M, M, K)
pi_shape = (M, K)
z_shape = (M, T, K)
zz_shape = (M, M, T - 1, K)
logZ_shape = (K,)
d_shape = (D, K)

A = np.random.rand(*A_shape)
pi = np.random.rand(*pi_shape)
z = np.random.rand(*z_shape)
zz = np.random.rand(*zz_shape)
logZ = np.random.rand(*logZ_shape)
dvec = np.random.rand(*d_shape)
dvec /= np.sum(dvec, 0, keepdims=True)

A_node = nd.DirichletNode(A, A, name='A')
pi_node = nd.DirichletNode(pi, pi, name='pi')
z_node = nd.MarkovChainNode(z, zz, logZ)
d_node = nd.DurationNode(M, dvec, None)

hmm = nd.HMMNode(z_node, A_node, pi_node)

def test_shapes_correct():
    assert_equals((M, T, K), (hmm.M, hmm.T,) + hmm.K)

def test_nodes_inserted():
    assert_true(hasattr(hmm, 'nodes'))
    assert_equals({'z', 'A', 'pi'}, set(hmm.nodes.keys()))

def test_entropy_is_scalar():
    assert_is_instance(hmm.entropy(), float)

def test_expected_log_prior_is_scalar():
    assert_is_instance(hmm.expected_log_prior(), float)

def test_update_works():
    psi = np.log(np.random.rand(T, M))
    flag = [0]
    def set_flag(x):
        flag[0] += 1
    hmm.update_finalizer = set_flag
    hmm.update(3, psi)
    assert_equals(flag[0], 1)
