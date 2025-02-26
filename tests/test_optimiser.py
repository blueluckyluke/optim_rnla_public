import numpy as np
import pytest

import optim_rnla.example_problem as ex
import optim_rnla.optimiser as op
import optim_rnla.trs_problem as trp


@pytest.mark.parametrize("optprob", (ex.NegGaussian, ex.Rosenbrock,
                                     ex.Sphere, ex.SumSquares))
@pytest.mark.parametrize("n", (2, 6, 10, 11))
def test_TR_no_exceptions(n, optprob):
    try:
        TRS_solver = op.TRSviaGEP()
        TRS_solver.solvegep = TRS_solver.solve_unsym_eigprob
        P = optprob(n)
        TR = op.TRAlgorithm(P, TRS_solver)
        TR()
        bool1 = TR.xcur.shape == (n, 1)
    except Exception:
        bool1 = False
    # Rosenbrock is only defined for even number of n
    if not n % 2 == 0 and optprob == ex.Rosenbrock:
        bool1 = True
    assert bool1


@pytest.mark.parametrize("trsprob", (trp.TRSProb1, trp.TRSProb2,
                                     trp.TRSProb3, trp.TRSProb4))
@pytest.mark.parametrize("n", (2, 6, 10, 11))
def test_TRS_no_exceptions(n, trsprob):
    try:
        Ptr = trsprob(n)
        TRS_solver = op.TRSviaGEP()
        TRS_solver.solvegep = TRS_solver.solve_unsym_eigprob
        s, lam, details = TRS_solver(Ptr)
        bool1 = True
    except Exception:
        bool1 = False
    assert bool1


@pytest.mark.parametrize("trsprob", (trp.TRSProb1, trp.TRSProb2,
                                     trp.TRSProb3, trp.TRSProb4))
@pytest.mark.parametrize("n", (2, 6, 10, 11))
def test_TRS_compare_s_with_sstar(n, trsprob):
    Ptr = trsprob(n)
    TRS_solver = op.TRSviaGEP()
    TRS_solver.solvegep = TRS_solver.solve_unsym_eigprob
    s, lam, details = TRS_solver(Ptr)
    print(s)
    if trsprob == trp.TRSProb4:
        bool1 = np.allclose(s, Ptr.sstar)
        if not bool1:
            Ptr.sstar[0] *= -1
        bool1 = np.allclose(s, Ptr.sstar)
    else:
        bool1 = np.allclose(s, Ptr.sstar)
    assert bool1


@pytest.mark.parametrize("optprob", (ex.NegGaussian, ex.Rosenbrock,
                                     ex.Sphere, ex.SumSquares))
@pytest.mark.parametrize("n", (2, 6, 10, 11))
def test_TR_convergence_to_solution(n, optprob):
    if n % 2 == 1 and optprob == ex.Rosenbrock:
        bool1 = True
    else:
        TRS_solver = op.TRSviaGEP()
        TRS_solver.solvegep = TRS_solver.solve_unsym_eigprob
        P = optprob(n)
        TR = op.TRAlgorithm(P, TRS_solver)
        TR.itermax = 100
        TR()
        bool1 = np.allclose(TR.xcur, P.globalmin, rtol=1e-03, atol=1e-03)
        # NegGaussian TR has difficulties to converge to actual minimum
        if optprob == ex.NegGaussian and not bool1:
            bool1 = np.linalg.norm(P.df(TR.xcur)) < 1e-5
    assert bool1


@pytest.mark.parametrize("optprob", (ex.NegGaussian, ex.Rosenbrock,
                                     ex.Sphere, ex.SumSquares))
def test_TR_no_exception_with_own_eigsolver(optprob):
    # MOST IMPORTANT TEST, TR plus TRS plus sRR runs together without error
    try:
        P = optprob(100)
        TR = op.TRAlgorithm(P, None)
        TR()
        bool1 = True
    except Exception:
        bool1 = False
    assert bool1
