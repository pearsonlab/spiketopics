"""
Define a constant node that is just a thin wrapper over a NumPy array.
"""
import numpy as np

class ConstNode(np.ndarray):
    def __new__(cls, arr):
        obj = np.asarray(arr).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def expected_x(self):
        return self

    def expected_log_x(self):
        return np.log(self)

    def update(self):
        pass

class ProductNode:
    def __init__(self, A, B):
        # assign parents
        self.A = A
        self.B = B
        self.shape = A.shape

    def expected_x(self):
        return self.A.expected_x() * self.B.expected_x()

    def expected_log_x(self):
        return self.A.expected_log_x() + self.B.expected_log_x()

    def update(self):
        pass
