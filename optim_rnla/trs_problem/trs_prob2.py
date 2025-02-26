import numpy as np
import scipy.sparse as sc

from .trsubproblem import TRSubproblem


class TRSProb2(TRSubproblem):
    """Boundary point solution with positive diagonal matrix H."""

    def __init__(self, n):
        self.n = n
        self.H = sc.diags(np.arange(1, self.n + 1), format="csr")
        self.lamstar = 10
        self.N = sc.identity(self.n, format="csr")
        self.sstar = np.ones((self.n, 1))
        self.g = (self.H + self.lamstar * self.N) @ self.sstar * (-1)
        self.delta = self.N_norm(self.sstar)
