import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import netsquid as ns
from fidelity_tests import bell_pair
import netsquid.components.instructions as instr
from netsquid.qubits.qubit import Qubit
from netsquid.components.qmemory import QuantumMemory
from netsquid.components import T1T2NoiseModel
ns.set_qstate_formalism(ns.qubits.DenseDMRepr)

class SwapSimulator():
    def __init__(self) -> None:
        self.mem = QuantumMemory("memB", num_positions=2)
        noiseModel = T1T2NoiseModel(T1=500, T2=50)
        for mem_pos in self.mem.mem_positions:
            mem_pos.models['noise_model'] = noiseModel
        self.reset()

    def reset(self):
        self.qA, qB_A = bell_pair()
        self.qC, qB_C = bell_pair()

        self.mem.put(qB_A, positions=0)
        self.mem.put(qB_C, positions=1)

    @property
    def qB_A(self) -> Qubit:
        return self.mem.peek(0)[0]
    
    @property
    def qB_C(self) -> Qubit:
        return self.mem.peek(1)[0]

    def swap(self):
        return instr.INSTR_MEASURE_BELL(self.mem, [0,1])
    
def simulateSwap():
    swapSim = SwapSimulator()
    ns.sim_run(duration=100)

    print("Time = ", ns.sim_time())
    print("Fidelity of A~B: ", ns.qubits.fidelity([swapSim.qA, swapSim.qB_A], ns.b00))
    print("Fidelity of B~C: ", ns.qubits.fidelity([swapSim.qC, swapSim.qB_C], ns.b00))

    print(swapSim.swap())
    print("Fidelity of A~C: ", ns.qubits.fidelity([swapSim.qA, swapSim.qC], ns.b00))

for _ in range(10):
    print("#"*15)
    simulateSwap()