import numpy as np
from scipy.linalg import sqrtm
import netsquid as ns

def manual_formula(dm1: np.ndarray, dm2: np.ndarray):
    sqrt_dm1 = sqrtm(dm1)
    assert isinstance(sqrt_dm1, np.ndarray) 

    prod = sqrt_dm1 @ dm2 @ sqrt_dm1
    sqrt_prod = sqrtm(prod)
    assert isinstance(sqrt_prod, np.ndarray) 

    return np.real(sqrt_prod.trace() ** 2)

def ketProduct_formula(idealState: np.ndarray, dm: np.ndarray):
    idealState_conjT = np.conj(idealState).T
    return np.real((idealState_conjT @ dm @ idealState)[0, 0])

def get_state_representations():
    ns.set_qstate_formalism(ns.qubits.qformalism.DenseDMRepr)
    q1, q2 = ns.qubits.create_qubits(2)
    ns.qubits.operate([q1], ns.H)
    ns.qubits.operate([q1,q2], ns.CNOT)
    dm_reference = ns.qubits.qubitapi.reduced_dm([q1,q2])

    p1, p2 = ns.qubits.create_qubits(2)
    p1.combine(p2)
    ns.qubits.operate(p1, ns.H)
    # ns.qubits.operate(p2, ns.H)
    # ns.qubits.operate([p1,p2], ns.CNOT)
    dm_combined = ns.qubits.qubitapi.reduced_dm([p1,p2])

    return dm_combined, dm_reference, ns.qubits.fidelity([p1,p2], ns.b00, squared=True)

def test_fidelity_manual_formula():
    dm_combined, dm_reference, nsFidelity = get_state_representations()
    print("(manual) Expected fidelity |00> vs |bell+> = ", manual_formula(dm_combined, dm_reference))
    print("(manual) Obtained fidelity |00> vs |bell+> = ", nsFidelity)

def test_fidelity_ketProduct():
    dm_combined, _, nsFidelity = get_state_representations()
    ketIdeal = ns.b00
    print("(ketProd) Expected fidelity |00> vs |bell+> = ", ketProduct_formula(ketIdeal, dm_combined))
    print("(ketProd) Obtained fidelity |00> vs |bell+> = ", nsFidelity)    

if __name__ == "__main__":
    test_fidelity_manual_formula()
    test_fidelity_ketProduct()