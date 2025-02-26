import numpy as np

from optim_rnla.trs_problem import TRSubproblem

from .trs_via_GEP import TRSviaGEP


class TRAlgorithm:
    """Apply trust region method to an optimisation problem.
    Call init(optim_prob) for setup where optim_prob
    is a instance of the OptimProb class.
    If needed algorithm parameters can be changed afterwards.
    __call__() performs the algorithm and stores the result in self.xcur.
    More details about the steps can be accessed via e.g. self.xlist.

    Algorithm reference:
    inspired by the book Trust Region Methods by
    Andrew R. Conn, Nicholas I. M. Gould, and Philippe L. Toint
    """

    def __init__(self, optim_prob, trsviagep=None):
        # set of parameters which define a specific TR method
        self.delta0 = 1
        self.deltamax = 100
        self.etav = 0.9  # very successful step
        self.gammav = 2  # radius of very succesful step is increased by that
        self.eta = 0.1  # threshold to still accept step
        self.gamma = 0.5  # if unsuccesful radius in decreased by that

        # convergence termination criterions
        self.itermax = 10
        self.grad_eps = 1e-5

        # unless specific TRS solver is handed over, default TRS solver is used
        if trsviagep:
            self.trs_solver = trsviagep
        else:
            self.trs_solver = TRSviaGEP()
        self.optim_prob = optim_prob
        self.n = optim_prob.n

        # initializes first step
        self.delta = self.delta0
        self.xcur = np.ones((self.n, 1))
        self.iter = 0

        self.xlist = [self.xcur]
        self.deltalist = [self.delta]
        self.rholist = [0]
        self.lamlist = [0]
        self.detailslist = ["empty"]

    def build_trs_model(self, *arwgs):
        self.TRS = TRSubproblem(self.optim_prob, self.xcur, self.delta, None)

    def solve_TRS(self):
        self.sstar, self.lamstar, self.details = self.trs_solver(self.TRS)
        self.sstar = self.sstar.reshape((-1, 1))

    def evaluate_TRS_sol(self):
        xnext = self.xcur + self.sstar
        try:
            rho = ((self.optim_prob.f(self.xcur) - self.optim_prob.f(xnext)
                    ) / ((-1) * self.TRS.eval(self.sstar)))[0, 0]
        except ZeroDivisionError:
            rho = 1
        if rho >= self.etav and self.lamstar > 0:
            self.delta *= self.gammav  # gammav >1
            self.xcur = xnext
        elif rho >= self.eta:
            self.xcur = xnext
        else:
            # Step is not accepted and trust radius is decreased
            # self.xcur=self.xcur
            self.delta *= self.gamma  # gamma < 1
        self.xlist.append(self.xcur)
        self.deltalist.append(self.delta)
        self.lamlist.append(self.lamstar)
        self.rholist.append(rho)
        self.detailslist.append(self.details)

    def stop_crit(self):
        """Returns True if another TRS iteration should be performed."""
        self.iter += 1
        if self.iter > self.itermax:
            print(f"itermax {self.itermax} was reached.")
            return False
        elif np.linalg.norm(self.optim_prob.df(self.xcur)) < self.grad_eps:
            print(
                f"TR method converged to point with norm of gradient {np.linalg.norm(self.optim_prob.df(self.xcur))}")  # noqa E501
            return False
        else:
            return True

    def __call__(self):
        """Execute the Trust region algorithm."""
        while self.stop_crit():
            self.build_trs_model()
            self.solve_TRS()
            self.evaluate_TRS_sol()
            print(
                f" After Iter {self.iter}; current x starts {self.xcur[:4,0]}; rho {self.rholist[-1]}; {self.details}; delta new {self.deltalist[-1]}"  # noqa E501
            )
