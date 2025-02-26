import numpy as np
import scipy.sparse as sc

from .trsubproblem import TRSubproblem


class TRSProb4(TRSubproblem):
    """HARD CASE! Creates a boundary point with
    diag matrix with largest negative eigenvalue -n."""

    def __init__(self, n):
        self.n = n
        diagtmp = np.arange(-n, n, 2)
        self.H = sc.diags(diagtmp, format="csr")
        self.N = sc.identity(self.n, format="csr")
        self.sstar = np.ones((self.n, 1))
        self.lamstar = self.n
        self.g = (self.H + self.lamstar * self.N) @ self.sstar * (-1)
        self.delta = self.N_norm(self.sstar)
