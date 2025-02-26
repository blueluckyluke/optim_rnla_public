import numpy as np
import scipy


def blockkrylov(M, B0=None, b=1, depth=10, partialorth=1, blockorth=True):
    """Build a block Krylov subspace for matrix M
    with initial block B0 and b additional random vectors.
    If blockorth, every block is orthogonalised to itself.
    If partial >0 then block is orthogonalised against last k blocks.
    Returns matrix for basis B and M*B
    """

    # build initial block
    n = M.shape[0]
    k = partialorth
    if B0 is not None:
        b0 = B0.shape[1]
        if b > 0:
            Bnext = np.concatenate((B0, np.random.randn(n, b)), axis=1)
        else:
            Bnext = B0
    else:
        b0 = 0
        Bnext = np.random.randn(n, b)
    if B0 is None and b == 0:
        raise ValueError("Block Krylov space generation failed due no initial vector(s). Try increasing dimension of Problem.")  # noqa E401

    b = b0 + b
    B = np.zeros((n, b * depth))
    MB = np.zeros((n, b * depth))
    Bnext, _ = scipy.linalg.qr(Bnext, mode="economic")
    B[:, :b] = Bnext

    # compute Krylov space
    for i in range(depth - 1):
        Bnext = M @ Bnext
        MB[:, i * b: (i + 1) * b] = Bnext
        ki = i - k + 1
        if ki < 0:
            ki = 0
        if k > 0:
            Bnext = Bnext - \
                B[:, ki * b: (i + 1) * b] @ (B[:, ki *
                                               b: (i + 1) * b].T @ Bnext)
        if blockorth:
            Bnext, _ = scipy.linalg.qr(Bnext, mode="economic")
        else:
            Bnext = Bnext / np.linalg.norm(Bnext, axis=0)
        B[:, (i + 1) * b: (i + 2) * b] = Bnext
    MB[:, b * (depth - 1):] = M @ Bnext
    return B, MB
