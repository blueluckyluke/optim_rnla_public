import numpy as np
import pytest

import optim_rnla.trs_problem as trp


@pytest.mark.parametrize("n", (3, 10, 100))
def test_trs_prob1(n):
    P = trp.TRSProb1(n)
    assert P.check_solution()


@pytest.mark.parametrize("n", (3, 10, 100))
def test_trs_prob2(n):
    P = trp.TRSProb2(n)
    assert P.check_solution()


def test_trs_prob2b():
    P = trp.TRSProb2(10)
    P.lamstar = 10000
    assert not P.check_solution()


@pytest.mark.parametrize("n", (9, 10))
def test_trs_prob3(n):
    P = trp.TRSProb2(n)
    assert P.check_solution()


@pytest.mark.parametrize("n", (3, 10, 100))
def test_trs_prob4(n):
    P = trp.TRSProb2(n)
    # since its the hard case includeing psd_check would fail
    assert P.check_solution(include_psd_check=False)


@pytest.mark.parametrize("n", (3, 10, 100))
def test_trs_prob4a(n):
    P = trp.TRSProb4(n)
    # eigvec corresponds to the eigenvector of the smallest eigenvalue
    eigvec = np.zeros((n, 1))
    eigvec[0] = 1
    assert (P.g.T @ eigvec)[0, 0] == 0


def test_setup_TRS_via_param():
    H = np.eye(10)
    g = np.ones((10, 1))
    P = trp.TRSubproblem(None, None, delta=10, mnorm=None, H=H, g=g, e="trash")
    if hasattr(P, "H") and hasattr(P, "g") \
            and hasattr(P, "delta") and hasattr(P, "N"):
        boolvar = True
    else:
        boolvar = False
    assert boolvar
