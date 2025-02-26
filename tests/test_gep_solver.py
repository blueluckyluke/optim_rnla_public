import numpy as np
import pytest
import scipy

from optim_rnla.basisgeneration import blockkrylov
from optim_rnla.gep_solver import GEPSolver


def M(n):
    """Build matrix with known eigenvalues."""
    Q, _ = scipy.linalg.qr(np.random.randn(n, n))
    dia = np.arange(-n, n, 2)
    dia[-1] = 2 * n
    return Q @ np.diag(dia) @ Q.T


@pytest.mark.parametrize("n", (100, 200))
@pytest.mark.parametrize("which", ("plain", "sketched"))
@pytest.mark.parametrize("mode", ("performance", "investigation"))
def test_check_result(n, which, mode):
    RR = GEPSolver(which, mode)
    B, MB = blockkrylov(M(n), B0=None, b=int(n / 25), depth=5, partialorth=1)
    lam, V = RR(MB, B)
    # these methods are not designed for high accuracy!
    if np.abs(lam - 2 * n) / (2 * n) < 1e-1 and V.shape == (n, 1):
        bool1 = True
    else:
        bool1 = False
    assert bool1
