import numpy as np
import scipy


class TRSubproblem:
    """Parent class to define Trust-region subproblem of the form:
    Find minimiser s (and Lagrange multiplier lam) with |s|_N<delta
    in N-matrix norm of g.T* s+ 1/2 s.T*H*s.If exact solution known
    it can be accessed via self.sstar and self.lamstar"""

    def __init__(self, optimprob, x, delta, mnorm=None, **kwargs):
        """Builds TRS based on optimprob. If optimprob is None,
        then tries to build TRS out of additional handed over
        parameters H, g and delta"""
        if optimprob:
            self.g = optimprob.df(x)
            self.H = optimprob.ddf(x)
            self.n = self.H.shape[1]
            self.delta = delta
            if mnorm:
                self.N = mnorm
            else:
                self.N = scipy.sparse.identity(self.n, format="csr")
        else:
            try:
                self.H = kwargs["H"]
                self.n = self.H.shape[1]
                self.g = kwargs["g"]
                self.delta = delta
                if mnorm:
                    self.N = mnorm
                else:
                    self.N = scipy.sparse.identity(self.n, format="csr")
            except KeyError:
                print("TRS could not be constructed.")

    def check_positive_def(self, A):
        try:
            scipy.linalg.cholesky(A.toarray())
        except scipy.linalg.LinAlgError:
            print("Matrix is not positive definite.")
            return False
        else:
            return True

    def N_norm(self, x):
        return np.sqrt(x.T @ self.N @ x)

    def eval(self, s):
        """Evaluate point s of the model."""
        return self.g.T @ s + 1 / 2 * s.T @ self.H @ s

    @property
    def solution_known(self):
        if hasattr(self, "sstar") and hasattr(self, "lamstar"):
            return True
        return False

    def check_solution(self, include_psd_check=True):
        """Check if solution fulfils 4 known optimality criterions
        and prints out errors."""
        if not self.solution_known:
            print("Solution not known.")
            return False

        normsstar = self.N_norm(self.sstar)

        if self.delta * (1 - 1e-10) > normsstar:
            bool1 = True
            region = "interior"
        elif self.delta * (1 - 1e-10) < normsstar and normsstar \
                < self.delta * (1 + 1e10):
            bool1 = True
            region = "on boundary"
        else:
            bool1 = False
            region = "outside"

        err2 = np.linalg.norm(
            (self.H + self.lamstar * self.N) @ self.sstar + self.g)
        if err2 < 1e-5:
            bool2 = True
        else:
            bool2 = False

        if abs(self.lamstar * (self.delta - normsstar)) < 1e-15:
            bool3 = True
        else:
            bool3 = False

        if include_psd_check:
            str4 = self.check_positive_def(self.H + self.lamstar * self.N)
        else:
            str4 = "NOT checked"

        if bool1 and bool2 and bool3 and str4:
            returnval = True
            valid = "VALID"
        else:
            valid = "NOT valid"
            returnval = False

        print(f"Solution of TRS is {valid}.")
        print(
            f"Solution point is {region}. Error is {err2}. Matrix psd?: {str4}")  # noqa E401

        return returnval
