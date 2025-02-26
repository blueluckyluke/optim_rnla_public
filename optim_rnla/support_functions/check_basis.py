import matplotlib.pyplot as plt
import numpy as np


def plot_basis(B, eps=1e-10):
    """Check orthogonality by plotting sparsity
    pattern of the matrix B.T * B."""
    BTB = B.T @ B
    BTB[np.abs(BTB) < eps] = 0
    plt.spy(BTB)
    plt.show()
