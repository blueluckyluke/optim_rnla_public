import numpy as np
import scipy.sparse as sc

from .optim_prob import OptimProb


class NegGaussian(OptimProb):
    """Implements a negative Gaussian function.

    f= - exp(- 1/2 * x.T * A * x)
    df= exp(- 1/2 * x.T * A * x) * A * x
    ddf =exp(- 1/2 * x.T * A * x) * (A - (A * x) * (x.T * A))
    """

    def __init__(self, n, density=0.8, seed=42, A=None):
        # Option to define the matrix A by hand.
        super().__init__(n)
        self.B = sc.rand(self.n, self.n, density, format="csr",
                         random_state=seed)
        # to assure A is positive semidefinite
        self.A = self.B @ self.B.T + 0.1 * sc.eye(n)
        # voids nan matries if dimension is small
        # if not self.A.nnz:
        #    self.A = sc.eye(n)
        if A is not None:
            self.A = sc.csr_matrix(A)

    def f(self, x):
        self.check_input_x(x)
        return (-1 * np.exp(-1 / 2 * x.T @ self.A @ x))[0, 0]

    def df(self, x):
        self.check_input_x(x)
        return -1 * self.f(x) * self.A @ x

    def ddf(self, x):
        self.check_input_x(x)
        Ax = self.A @ x
        # potential speed-up via defining an operator
        return self.f(x) * (self.A - Ax @ Ax.T)

    @property
    def globalmin(self):
        return np.zeros((self.n, 1))
