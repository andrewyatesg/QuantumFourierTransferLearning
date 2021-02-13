import numpy as np
import functools

X = np.array([[0, 1],
              [1, 0]])
Y = np.array([[0, -1j],
              [1j, 0]])
Z = np.array([[1, 0],
              [0, -1]])


def operator_on_qubit(op, i, N):
    ops = [np.identity(2) for _ in range(N)]
    ops[i] = op
    return functools.reduce(np.kron, ops)


def hamiltonian_mat(J, h1, h2, N):
    acc1 = 0
    for i in range(N-2):
        acc1 = acc1 + operator_on_qubit(Z, i, N) @ operator_on_qubit(X, i+1, N) @ operator_on_qubit(Z, i+2, N)
    acc1 = acc1

    acc2 = 0
    for i in range(N):
        acc2 = acc2 + operator_on_qubit(X, i, N)
    acc2 = acc2

    acc3 = 0
    for i in range(N-1):
        acc3 = acc3 + operator_on_qubit(X, i, N) @ operator_on_qubit(X, i+1, N)
    acc3 = acc3

    return -J * acc1 - h1 * acc2 - h2 * acc3


def all_grnd_states(hamiltonian):
    eig_vals, eig_vecs = np.linalg.eig(hamiltonian)
    sort_perm = eig_vals.argsort()
    eig_vals.sort()
    eig_vecs = eig_vecs[:, sort_perm]

    min = np.amin(eig_vals)
    grnd_idxs = np.argwhere(eig_vals == min).flatten()
    return min, eig_vecs[:, grnd_idxs]


# print(all_grnd_states(hamiltonian_mat(1.2, 5.3, 10, 5)))
