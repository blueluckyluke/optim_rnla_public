from numbers import Number

import numpy as np
import pytest
import scipy.sparse as sc

import optim_rnla.example_problem as ex


@pytest.mark.parametrize("n", (1, 10, 100, 1000))
def test_sphere_init(n):
    optim = ex.Sphere(n)
    assert optim.n == n


def test_sphere_f():
    optim = ex.Sphere(5)
    x = np.array([[1, 1, 1, 1, 1]]).T
    assert optim.f(x) == 5


def test_sphere_df():
    optim = ex.Sphere(5)
    x = np.array([[1, 1, 1, 1, 1]]).T
    sol = np.array([[2, 2, 2, 2, 2]]).T
    assert (optim.df(x) == sol).all()


def test_sphere_ddf():
    optim = ex.Sphere(5)
    x = np.array([[1, 1, 1, 1, 1]]).T
    sol = sol = np.array([[2, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 2, 0, 0],
                          [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])
    assert (optim.ddf(x) == sol).all()


def test_sumsquares_f():
    P = ex.SumSquares(5)
    x = np.array([[1, 1, 1, 1, 1]]).T
    assert P.f(x) == 15


def test_sumsquares_df():
    P = ex.SumSquares(5)
    x = np.array([[1, 1, 1, 1, 1]]).T
    sol = np.array([[2, 4, 6, 8, 10]]).T
    assert (P.df(x) == sol).all()


def test_sumsquares_ddf():
    P = ex.SumSquares(5)
    x = np.array([[1, 1, 1, 1, 1]]).T
    sol = np.array([[2, 0, 0, 0, 0], [0, 4, 0, 0, 0], [
                   0, 0, 6, 0, 0], [0, 0, 0, 8, 0], [0, 0, 0, 0, 10]])
    assert (P.ddf(x).toarray() == sol).all()


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_rosenbrock_f1(n):
    P = ex.Rosenbrock(n)
    x = np.ones((n, 1))
    # global minimum at x= [1,1,....1] with value 1
    assert P.f(x) == 0


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_rosenbrock_f0(n):
    P = ex.Rosenbrock(n)
    x = np.zeros((n, 1))
    assert P.f(x) == n / 2


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_rosenbrock_df1(n):
    P = ex.Rosenbrock(n)
    x = np.ones((n, 1))
    # global minimum at x= [1,1,....1] with value 1
    assert (P.df(x) == np.zeros((n, 1))).all()


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_rosenbrock_df0(n):
    P = ex.Rosenbrock(n)
    x = np.zeros((n, 1))
    sol = np.zeros((n, 1))
    sol[::2, :] = -2
    assert (P.df(x) == sol).all()


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_rosenbrock_ddf0(n):
    P = ex.Rosenbrock(n)
    x = np.zeros((n, 1))
    sol = np.zeros(n)
    sol[::2] = 2
    sol[1::2] = 200
    ind = np.arange(0, n)
    M = sc.csr_matrix((sol, (ind, ind)))
    assert (P.ddf(x).toarray() == M.toarray()).all()


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_rosenbrock_ddf1(n):
    P = ex.Rosenbrock(n)
    x = np.ones((n, 1))
    assert np.sum(np.diag(P.ddf(x).toarray())) == 1002 * n / 2


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_neggaussian_f(n):
    A = np.eye(n)
    P = ex.NegGaussian(n, A=A)
    x = np.ones((n, 1))
    assert isinstance(P.f(x), Number)


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_neggaussian_df(n):
    A = np.eye(n)
    P = ex.NegGaussian(n, A=A)
    x = np.zeros((n, 1))
    assert (P.df(x) == x).all()


@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_neggaussian_ddf(n):
    A = np.eye(n)
    P = ex.NegGaussian(n, A=A)
    x = np.ones((n, 1))
    assert np.diag(P.ddf(x)) == pytest.approx(np.zeros(n))
