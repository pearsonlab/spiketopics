"""
Define a constant node that is just a thin wrapper over a NumPy array.
"""
import numpy as np

class ConstNode(np.ndarray):
    def expected_x(self):
        return self

    def expected_log_x(self):
        return np.log(self)

class ProductNode:
    def __init__(self, A, B):
        # assign parents
        self.A = A
        self.B = B

    def expected_x(self):
        return self.A.expected_x() * self.B.expected_x()

    def expected_log_x(self):
        return self.A.expected_log_x() + self.B.expected_log_x()
