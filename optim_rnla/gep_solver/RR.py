import numpy as np
import scipy.fft as fft
import scipy.linalg as la


class GEPSolver:
    """Class to contain sketched and plain Rayleigh-Ritz eigensolver
    algorithms. In __init__(which,mode) which offers options sketched and plain
    mode offers performance or investigation (later one saves all eigen pairs
    of basis under self.lams and self.vecs)
    For sketching it is assumed B.shape=MB.shape=(n,r) n>>r (if n<4*r then
    appears ValueError: illegal value in 7th argument of internal geqrf)

    Reference for sketched Rayleigh-Ritz:
    Fast and Accurate Randomized Algorithms for Linear Systems and
    Eigenvalue Problems
    Yuji Nakatsukasa, Joel A. Tropp
    """

    def __init__(self, which="sketched", mode="performance"):
        if which == "sketched":
            self.solver = self.sketchedRR
        elif which == "plain":
            self.solver = self.plainRR
        else:
            raise NotImplementedError

        if mode == "performance":
            self.modesave = False
        elif mode == "investigation":
            self.modesave = True
        else:
            raise NotImplementedError

    def __call__(self, MB, B):
        """Return rightmost lamda and according eigenvector."""
        return self.solver(MB, B)

    def plainRR(self, MB, B):
        """Return largest real eigenpair in basis."""
        Q, R = la.qr(B, mode="economic")
        lams, V = la.eig(la.inv(R) @ Q.T @ MB)
        return self.postprocess(lams, V, B)

    def sketchedRR(self, MB, B):
        n = MB.shape[0]
        # SRFT (efficient alternative to gaussian matrices)
        signs = np.sign(np.random.randn(n, 1))
        # indx = np.random.randint(0, n, 4*MB.shape[1],) no replace
        if MB.shape[1]*4 > MB.shape[0]:
            raise ValueError(f"For sketching MB.shape[1]*4 must be smaller than MB.shape[0], {MB.shape[1]*4} <? {MB.shape[0]}")  # noqa E401
        indx = np.random.choice(
            np.arange(0, n), size=4 * MB.shape[1], replace=False)
        # includes normalisation
        SMBtmp = fft.dct(signs * MB, axis=0, norm="ortho")
        SMBtmp[0, :] /= np.sqrt(2)
        SBtmp = fft.dct(signs * B, axis=0, norm="ortho")
        SBtmp[0, :] /= np.sqrt(2)
        SMB = SMBtmp[indx, :]
        SB = SBtmp[indx, :]
        Q, R = la.qr(SB, mode="economic")
        K = la.inv(R) @ Q.T @ SMB
        lams, V = la.eig(K)
        return self.postprocess(lams, V, B)

    def postprocess(self, lams, V, B):
        if self.modesave:
            self.lams = lams
            self.vecs = B @ V
        lam = np.max(np.real(lams))
        ind = np.argmax(np.real(lams))
        vec = V[:, ind: ind + 1]
        Bvec = B @ vec
        return lam, Bvec
