import numpy as np
from scipy.linalg import cholesky, solve_triangular
from copy import deepcopy


def is_symmetric(A):
    """Check if matrix A is symmetric (A == A^T)."""
    return np.allclose(A, A.T)


def is_positive_definite(A):
    """Check if matrix A is positive definite (all eigenvalues > 0)."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky decomposition."""
    L = cholesky(A, lower=True)  # Compute L such that A = L * L^T
    y = solve_triangular(L, b, lower=True)  # Solve L * y = b
    x = solve_triangular(L.T, y, lower=False)  # Solve L^T * x = y
    return x


def lu_factorization(A):
    """Perform LU factorization using Doolittle's method."""
    n = len(A)
    L = np.eye(n)
    U = deepcopy(A)

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j] = U[j] - factor * U[i]

    return L, U


def lu_solve(A, b):
    """Solve Ax = b using Doolittle LU Factorization."""
    L, U = lu_factorization(A)

    # Forward substitution for L * y = b
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(len(L)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Back substitution for U * x = y
    x = np.zeros_like(y, dtype=np.float64)
    for i in range(len(U) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def main():
    systems = [
        (np.array([[1, -1, 3, 2], [-1, 5, -5, -2], [3, -5, 19, 3], [2, -2, 3, 21]]), np.array([15, -35, 94, 1])),
        (np.array([[4, 2, 4, 0], [2, 2, 3, 2], [4, 3, 6, 3], [0, 2, 3, 9]]), np.array([20, 36, 60, 122]))
    ]

    for i, (A, b) in enumerate(systems):
        print(f"\nSystem {i + 1}:")
        if is_symmetric(A) and is_positive_definite(A):
            x = cholesky_solve(A, b)
            method = "Cholesky decomposition"
        else:
            x = lu_solve(A, b)
            method = "Doolittle LU Factorization"

        print(f"Solution: {np.round(x, 4)}")
        print(f"Method used: {method}")


if __name__ == "__main__":
    main()
