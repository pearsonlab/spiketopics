"""
Tests for constant node.
"""
from __future__ import division
from nose.tools import assert_equals, assert_is_instance, assert_raises, assert_true, assert_in, assert_not_in, assert_is, set_trace 
import numpy as np
import numpy.testing as npt
from spiketopics.nodes.ConstNode import ConstNode

def test_can_make_node():
    arr = np.random.rand(5, 7)
    cn = arr.view(ConstNode)
    npt.assert_array_equal(arr, cn.expected_x())