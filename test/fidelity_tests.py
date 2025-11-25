from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from directComm import *
import netsquid.qubits.qubitapi as qapi
from netsquid.components import T1T2NoiseModel

def state(q1):
    print("State: \n", ns.qubits.reduced_dm(q1))
def fidelity(q1,q2):
    print(f"Fidelity = {qapi.fidelity([q1,q2], ns.b00)}")
def bell_pair():
    q1, q2 = ns.qubits.create_qubits(2)
    ns.qubits.operate([q1], ns.H)
    ns.qubits.operate([q1,q2], ns.CNOT)
    return q1,q2

def test_manual_states():
    q1,q2 = bell_pair()
    state(q1)
    fidelity(q1,q2)

    ns.qubits.apply_dda_noise([q2], depol=0.8)

    state(q1)
    fidelity(q1,q2)

def test_noise_model(t, t1, t2, Model=T1T2NoiseModel):
    print(f"T1T2 Noise model test w/ t = {t} ; (t1,t2)=({t1},{t2})")
    errModel = Model(t1, t2)
    q1,q2 = bell_pair()

    errModel.apply_noise(q2, t)
    state(q2)
    fidelity(q1,q2)

def test_sim_noise_range(dist, rates):
    for t1,t2 in rates:
        results = setup_sim(shots=200, distance=dist, p_loss_init=0.3, t1=t1, t2=t2)
        fidelities = [r[3] for r in results]
        unique_fids = set(fidelities)
        print(f"(t1,t2)=({t1/1000},{t2/1000})us. Num diff. fidelities =", len(unique_fids), f" ; {unique_fids}")

def main():
    ns.set_qstate_formalism(ns.qubits.qformalism.DenseDMRepr)
    print(ns.get_qstate_formalism())
    # test_noise_model(1, 0, 0)
    # test_noise_model(5, 0, 0)
    dist = 10
    C = 2e5 / 1e9 # km/ns
    travelTime_ns = dist/C
    test_sim_noise_range(dist, 
        [(0,0), 
         (1e0*travelTime_ns, 1e-1*travelTime_ns), 
         (5e0*travelTime_ns, 5e-1*travelTime_ns), 
         (1e1*travelTime_ns, 1e0*travelTime_ns), 
         (1e1*travelTime_ns, 1e1*travelTime_ns), 
         (1e2*travelTime_ns, 1e1*travelTime_ns), 
         (1e3*travelTime_ns, 1e2*travelTime_ns), 
        ])

if __name__ == "__main__":
    main()