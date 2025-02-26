import numpy as np
import scipy.sparse as sc

from optim_rnla.basisgeneration import blockkrylov
from optim_rnla.gep_solver import GEPSolver


class TRSviaGEP:
    """Implement a specific trust region subproblem solver.
    Call __call__(trsubproblem) to solve trsubproblem, of type TRSubproblem.
    Returns solution vector (ndarray), lambda (float) and details (string).

    Reference:
    Solving the trust-region subproblem by a generalized eigenvalue problem
    Satoru Adachi , Satoru Iwata , Yuji Nakatsukasa , and Akiko Takeda
    """

    def __init__(self):

        self.tol_cg = 1e-5
        self.tolhardcase = 1e-4
        self.whichgepsolver = "sketched"
        self.modegepsolver = "performance"
        # self.solvegep = self.solve_unsym_eigprob
        self.solvegep = self.solve_via_gepsolver

    def __call__(self, trsubproblem):
        self.trsp = trsubproblem
        self.cg_for_interior_sol()
        self.setup_gep(as_operator=False)
        self.solvegep()
        self.postprocess()
        self.hardcase()
        return self.compare_s0_s1()

    def cg_for_interior_sol(self):
        """Find potential interior solutioin via conjugate gradient.
        Since problem might be indefinite CG can fail or raise warnings.
        """
        s0, self.exit_codecg = sc.linalg.cg(
            self.trsp.H, -self.trsp.g, tol=self.tol_cg, atol=0)
        s0 = s0.reshape((-1, 1))
        if (
            self.exit_codecg >= 0
            and np.linalg.norm(self.trsp.H @ s0 + self.trsp.g) /
            np.linalg.norm(self.trsp.g) < self.tol_cg
        ):
            if self.trsp.N_norm(s0) <= self.trsp.delta:
                self.s0 = s0
            else:
                self.s0 = None
        else:
            self.s0 = None

    def setup_gep(self, as_operator=True):
        """Set up generalized eigenvalue problem of this form.
        Either as sparse operator or as dense matrix.
        M*v=lam*J*v
        M= [[-N, H],[H, -g*g.T/delta**2]]
        J = [[0,-N],[-N,0]]
        """
        if as_operator:

            def Moptimesv(v):
                try:
                    if v.shape[1] == 1:
                        pass
                except IndexError:
                    v = v.reshape((-1, 1))
                v1 = v[: self.trsp.n]
                v2 = v[self.trsp.n:]
                y1 = -self.trsp.N @ v1 + self.trsp.H @ v2
                y2 = self.trsp.H @ v2 - self.trsp.g * \
                    np.vdot(self.trsp.g, v2) / (self.trsp.delta**2)
                return np.concatenate((y1, y2), axis=0)

            self.M = sc.linalg.LinearOperator(
                (2 * self.trsp.n, 2 * self.trsp.n), matvec=Moptimesv)
        else:
            self.M = sc.bmat(
                [[-self.trsp.N, self.trsp.H],
                 [self.trsp.H,
                 - self.trsp.g @ self.trsp.g.T / (self.trsp.delta**2)]],
                format="csr",
            )
        self.J = sc.bmat(
            [[None, -self.trsp.N], [-self.trsp.N, None]], format="csr")
        self.Jinv = sc.linalg.inv(self.J)
        if as_operator:

            def Jtimesv(v):
                try:
                    if v.shape[1] == 1:
                        pass
                except IndexError:
                    v = v.reshape((-1, 1))
                return self.Jinv @ v
            self.Jinvop = sc.linalg.LinearOperator(
                (2*self.trsp.n, 2*self.trsp.n), matvec=Jtimesv)
            self.K = self.Jinvop @ self.M
        else:
            self.K = self.Jinv @ self.M

    def solve_via_gepsolver(self):
        """Solve eigenvalue problem via (sketched) Rayleigh-Ritz."""
        # self.K = sc.linalg.inv(self.J) @ self.M
        self.gepsolver = GEPSolver(self.whichgepsolver, self.modegepsolver)
        self.B, self.KB = blockkrylov(self.K, B0=None, b=int(
            self.trsp.n / 25), depth=5, partialorth=1)
        self.lam, self.vec = self.gepsolver(self.KB, self.B)

    def solve_unsym_eigprob(self):
        """Default scipy function to solve eigenproblem."""
        # self.setup_gep(as_operator=False)
        # self.K = sc.linalg.inv(self.J) @ self.M
        self.lam, self.vec = sc.linalg.eigs(self.K, k=1, which="LR")

    # Following does not work since scipy can only
    # solve GEP if either M or J are pd
    # def solve_gep(self, opt="scipy"):
    #   self.lam, self.vec = sc.linalg.eigsh(self.M, 1, self.J, which="LA")

    def postprocess(self):
        if np.linalg.norm(np.real(self.vec)) < 1e-3:
            self.vec = np.imag(self.vec)
        else:
            self.vec = np.real(self.vec)
        self.lam = np.real(self.lam)
        self.v1 = np.real(self.vec[: self.trsp.n, :])
        self.v2 = np.real(self.vec[self.trsp.n:, :])
        try:
            self.lam = np.real(self.lam[0])
        except IndexError:
            self.lam = np.real(self.lam)
        self.s1 = -(np.sign(self.trsp.g.T @ self.v2)
                    )[0, 0] * self.trsp.delta * self.v1 /\
            self.trsp.N_norm(self.v1)
        self.details_bound = "boundary"

    def hardcase(self):
        # code for hard case (currently a simplified version)
        if self.trsp.N_norm(self.v1) < self.tolhardcase:
            # set up and solve Htilde * q=-g
            #  with Htilde= H + lamstar * N + lamstar * N * v2 * v2.T * N
            # Htilde by construction positive definite
            def Htildetimesp(p):
                try:
                    if p.shape[1] == 1:
                        pass
                except IndexError:
                    p = p.reshape((-1, 1))
                y = self.trsp.H @ p + self.lam * self.trsp.N @ p
                y = y + self.lam * (self.trsp.N @ self.v2) * \
                    (self.v2.T @ self.trsp.N @ p)
                return y

            self.Htilde = sc.linalg.LinearOperator(
                (self.trsp.n, self.trsp.n), matvec=Htildetimesp)
            self.q, exit_code = sc.linalg.cg(
                self.Htilde, -self.trsp.g, tol=self.tol_cg, atol=0)
            self.q = self.q.reshape((-1, 1))

            # find scalar value eta s.t. ||q+ eta*v2||_N=delta
            a = self.v2.T @ self.v2
            b = 2 * self.v2.T @ self.trsp.N @ self.q
            c = self.q.T @ self.trsp.N @ self.q - self.trsp.delta**2
            eta = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            self.s1 = self.q + eta * self.v2
            self.details_bound = "HARD CASE"

    def compare_s0_s1(self):
        if self.s0 is not None:
            if self.trsp.eval(self.s0) < self.trsp.eval(self.s1):
                return self.s0, 0, "interior"
        return self.s1, self.lam, self.details_bound
