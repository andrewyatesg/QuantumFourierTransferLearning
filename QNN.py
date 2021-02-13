import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines import ArbitraryUnitary

dev = qml.device('qiskit.aer', wires=10)

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def unitary_layer(all_wires, params, k = 2):
    """
    Layers of unitaries
    The first and last qubits are ancilla qubits
    Input:
        all_wires: list of wires to be used
        params: parameters for unitaries
        k: every kth qubit, we apply an aribtrary unitary gate to k+1 qubit
    """
    qml.broadcast(ArbitraryUnitary, wires= all_wires, pattern= "double", parameters = params)

def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

def ansatz(n_qubits):
    H_layer(n_qubits)
    unitary_layer(range(n_qubits), np.random.randn(15*(n_qubits/2)))
    entangling_layer(nqubits)
    unitary_layer(range(n_qubits), np.random.randn(15*(n_qubits/2)))
    entangling_layer(nqubits)

@qml.qnode(dev)
def qNN(n_qubits):
    params = get_parameters(n_qubits)
    ansatz(n_qubits)
    return qml.expval(qml.PauliZ(0))
