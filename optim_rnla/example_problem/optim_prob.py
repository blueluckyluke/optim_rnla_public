from abc import ABC


class OptimProb(ABC):
    """Parent class for optimisation problem.
    Point x should be a numpy array of shape (n,1).
    df returns a vector of shape (n,1)."""

    def __init__(self, n):
        self.n = n

    def f(self, x):
        raise NotImplementedError

    def df(self, x):
        raise NotImplementedError

    def ddf(self, x):
        raise NotImplementedError

    def check_input_x(self, x):
        if x.shape != (self.n, 1):
            raise ValueError(
                f"Expected argument of shape {(self.n,1)}, \
                  got instead shape {x.shape}"
            )

    @property
    def globalmin(self):
        raise NotImplementedError
