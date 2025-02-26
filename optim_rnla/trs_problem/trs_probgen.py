import numpy as np
import scipy.sparse as sc

from .trsubproblem import TRSubproblem


class TRSProbGen(TRSubproblem):
    """More Generic framework for TRS.
    Defines a problem with boundary solution."""

    def __init__(self, n, H=None, opt="randgaus"):
        self.n = n
        self.H = H
        if opt == "randgaus":
            self.H = np.random.normal(size=(n, n)) * 10
            self.H = sc.csr_matrix(self.H + self.H.T)
        self.lamstar = 10
        self.N = sc.identity(self.n, format="csr")
        self.sstar = np.ones((self.n, 1))
        self.g = (self.H + self.lamstar * self.N) @ self.sstar * (-1)
        self.delta = self.N_norm(self.sstar)
