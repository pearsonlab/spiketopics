"""
Tests for constant node.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace 
import numpy as np
import numpy.testing as npt
from spiketopics.nodes.utility_nodes import *

def test_can_make_node():
    arr = np.random.rand(5, 7)
    cn = arr.view(ConstNode)
    npt.assert_array_equal(arr, cn.expected_x())
    npt.assert_array_equal(np.log(arr), cn.expected_log_x())

def test_product_node():
    arr1 = np.random.rand(5, 7)
    arr2 = np.random.rand(5, 7)
    cn1 = arr1.view(ConstNode)
    cn2 = arr2.view(ConstNode)
    pn = ProductNode(cn1, cn2)
    npt.assert_array_equal(pn.expected_x(), arr1 * arr2)
    npt.assert_array_equal(pn.expected_log_x(), np.log(arr1) + 
        np.log(arr2))
