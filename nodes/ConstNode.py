"""
Define a constant node that is just a thin wrapper over a NumPy array.
"""
import numpy as np

class ConstNode(np.ndarray):
    def expected_x(self):
        return self