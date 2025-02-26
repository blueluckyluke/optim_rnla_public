import numpy as np
import scipy.sparse as sc

from .optim_prob import OptimProb


class SumSquares(OptimProb):
    """Parabolic test problem
    f= sum_{i=0}^n i* (x_i)^2
    df_i =2* x_i * i
    ddf_ij= kroneckerdelta_ij *i"""

    def f(self, x):
        self.check_input_x(x)
        tmp = 0
        for i, x_i in enumerate(x):
            tmp += (i + 1) * x_i**2
        return float(tmp[0])

    def df(self, x):
        self.check_input_x(x)
        tmp = np.zeros((self.n, 1))
        for i, x_i in enumerate(x):
            tmp[i] = 2 * (i + 1) * x_i
        return tmp

    def ddf(self, x):
        self.check_input_x(x)
        data = np.arange(2, 2 * self.n + 1, 2)
        inddata = np.arange(0, self.n)
        return sc.csr_matrix((data, (inddata, inddata)))

    @property
    def globalmin(self):
        return np.zeros((self.n, 1))
