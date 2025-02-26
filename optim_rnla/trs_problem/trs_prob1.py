import numpy as np
import scipy.sparse as sc

from .trsubproblem import TRSubproblem


class TRSProb1(TRSubproblem):
    """Interior point solution with positive definite diagonal matrix.
    n number of dimensions"""

    def __init__(self, n):
        self.n = n
        self.H = sc.diags(np.arange(1, self.n + 1), format="csr")
        self.lamstar = 0
        self.N = sc.identity(self.n, format="csr")
        self.sstar = np.ones((self.n, 1))
        self.g = self.H @ self.sstar * (-1)
        self.delta = 1.5 * self.N_norm(self.sstar)
        self.lamstar = 0
