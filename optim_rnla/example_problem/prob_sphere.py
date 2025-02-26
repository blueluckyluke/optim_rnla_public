import numpy as np
import scipy.sparse as sc

from .optim_prob import OptimProb


class Sphere(OptimProb):
    "Simple parabolic test problem."
    # f= sum_0^n(x_i)^2
    # df_i =2* x_i
    # ddf_ij= 2 * kroneckerdelta_ij

    def f(self, x):
        self.check_input_x(x)
        return np.sum(np.multiply(x, x))

    def df(self, x):
        self.check_input_x(x)
        return 2 * x

    def ddf(self, x):
        self.check_input_x(x)
        return sc.eye(self.n, format="csr") * 2

    @property
    def globalmin(self):
        return np.zeros((self.n, 1))
