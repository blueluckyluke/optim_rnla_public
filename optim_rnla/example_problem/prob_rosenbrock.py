import numpy as np
import scipy.sparse as sc

from .optim_prob import OptimProb


class Rosenbrock(OptimProb):
    """Implements the simple multidimensional extension of 
    Rosenbrock function.

    f= sum_{i=0}^n/2 (100*(x_{2i-1}**2-x_{2i})**2+ (x_{2i-1}-1)**2)
    """

    def __init__(self, n):
        if n % 2 != 0:
            raise ValueError("Dimension needs to be even.")
        super().__init__(n)

    def f(self, x):
        self.check_input_x(x)
        tmp = 0
        for i in range(int(self.n / 2)):
            tmp += 100 * (x[2 * i] ** 2 - x[2 * i + 1]
                          ) ** 2 + (x[2 * i] - 1) ** 2
        return float(tmp[0])

    def df(self, x):
        self.check_input_x(x)
        tmp = np.zeros((self.n, 1))
        for i in range(int(self.n / 2)):
            tmp[2 * i] = 400 * x[2 * i] ** 3 - 400 * \
                x[2 * i] * x[2 * i + 1] + 2 * x[2 * i] - 2
            tmp[2 * i + 1] = 200 * (x[2 * i + 1] - x[2 * i] ** 2)
        return tmp

    def ddf(self, x):
        self.check_input_x(x)

        datadiag = np.zeros((self.n,))
        datadiag[1::2] = 200
        datadiag[::2] = 1200 * x[::2, 0] ** 2 - 400 * x[1::2, 0] + 2
        inddiag = np.arange(0, self.n)

        dataoffdiag = -400 * x[::2, 0]
        indoffdiaga = np.arange(0, self.n, 2)
        indoffdiagb = np.arange(1, self.n, 2)
        H = sc.csr_matrix((datadiag, (inddiag, inddiag)))
        H += sc.csr_matrix((dataoffdiag, (indoffdiaga, indoffdiagb)),
                           shape=(self.n, self.n)) + sc.csr_matrix(
            (dataoffdiag, (indoffdiagb, indoffdiaga)), shape=(self.n, self.n)
        )
        return H

    @property
    def globalmin(self):
        return np.ones((self.n, 1))
