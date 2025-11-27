import numpy as np
from scipy.linalg import sqrtm
import netsquid as ns

def expected_fidelity(dm1: np.ndarray, dm2: np.ndarray):
    sqrt_dm1 = sqrtm(dm1)
    assert isinstance(sqrt_dm1, np.ndarray) 

    prod = sqrt_dm1 * dm2 * sqrt_dm1
    sqrt_prod = sqrtm(prod)
    assert isinstance(sqrt_prod, np.ndarray) 

    return sqrt_prod.trace() ** 2

import netsquid.qubits.dmtools

def test_fidelity():
    ns.set_qstate_formalism(ns.qubits.qformalism.DenseDMRepr)
    q1, q2 = ns.qubits.create_qubits(2)
    ns.qubits.operate([q1], ns.H)
    ns.qubits.operate([q1,q2], ns.CNOT)
    dm_reference = ns.qubits.qubitapi.reduced_dm([q1,q2])
    print(dm_reference)

    p1, p2 = ns.qubits.create_qubits(2)
    p1.combine(p2)
    ns.qubits.operate(p1, ns.H)
    # ns.qubits.operate(p2, ns.H)
    ns.qubits.operate([p1,p2], ns.CNOT)
    dm_combined = ns.qubits.qubitapi.reduced_dm([p1,p2])

    print(dm_combined)

    print("Expected fidelity |00> vs |bell+> = ", expected_fidelity(dm_combined, dm_reference))
    print("Obtained fidelity |00> vs |bell+> = ", ns.qubits.fidelity([p1,p2], ns.b00, squared=True))


if __name__ == "__main__":
    test_fidelity()