import numpy as np
import pytest
import scipy.sparse as sc

import optim_rnla.basisgeneration as bg


def M(n):
    return sc.rand(n, n, density=0.3, format="csr")


def B00(n, b0):
    return np.eye(n, b0)


@pytest.mark.parametrize("n", (100,))
@pytest.mark.parametrize("B0", (None, B00(100, 5)))
@pytest.mark.parametrize("b", (0, 1, 5))
@pytest.mark.parametrize("depth", (1, 10))
@pytest.mark.parametrize("partialorth", (0, 1, 5))
@pytest.mark.parametrize("blockorth", (True, False))
def test_no_error_plus_B_equal_MB(n, B0, b, depth, partialorth, blockorth):
    if B0 is None and b == 0:
        # this combination of parameters cannot work
        bool1 = True
    else:
        M0 = M(n)
        B, MB = bg.blockkrylov(M0, B0, b, depth, partialorth, blockorth)
        bool1 = np.allclose(M0 @ B, MB)
    assert bool1
