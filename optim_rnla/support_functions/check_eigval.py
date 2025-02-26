import matplotlib.pyplot as plt
import numpy as np


# Show fluctuating eigenvalues since scipy cannot solve this GEP
""" def plot_gep(trsviagep):
    t = trsviagep
    lams, vecs = sc.linalg.eigsh(t.M, t.trsp.n, t.J)
    # scatter plot of eigenvalues
    plt.scatter(np.real(lams), np.imag(lams))
    plt.show() """


def plot_eigval(K, asnumpyarray=False):
    if not asnumpyarray:
        K = K.toarray()
    lams, vecs = np.linalg.eig(K)
    plt.scatter(np.real(lams), np.imag(lams))
    plt.show()
