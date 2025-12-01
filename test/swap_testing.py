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
        self.mem = QuantumMemory("mem", num_positions=4)
        noiseModel = T1T2NoiseModel(T1=500, T2=50)
        for mem_pos in self.mem.mem_positions:
            mem_pos.models['noise_model'] = noiseModel
        self.reset()

    def reset(self):
        qA, qB_A = bell_pair()
        qC, qB_C = bell_pair()

        self.mem.put(qB_A, positions=0)
        self.mem.put(qB_C, positions=1)
        self.mem.put(qA, positions=2)
        self.mem.put(qC, positions=3)

    @property
    def qB_A(self) -> Qubit:
        return self.mem.peek(0)[0]
    
    @property
    def qB_C(self) -> Qubit:
        return self.mem.peek(1)[0]
    
    @property
    def qA(self) -> Qubit:
        return self.mem.peek(2)[0]
    
    @property
    def qC(self) -> Qubit:
        return self.mem.peek(3)[0]

    def swap(self):
        return instr.INSTR_MEASURE_BELL(self.mem, [0,1])[0] #type:ignore
    
    def correct(self, m):
        if m==1: instr.INSTR_X(self.mem, [3])
        if m==2: instr.INSTR_Y(self.mem, [3]) # NOTE: bell measure = 2 means A~C == b11
        if m==3: instr.INSTR_Z(self.mem, [3]) # NOTE: bell measure = 3 means A~C == b10
    
def simulateSwap():
    swapSim = SwapSimulator()
    ns.sim_run(duration=50)

    print("Time = ", ns.sim_time())
    fid1 = ns.qubits.fidelity([swapSim.qA, swapSim.qB_A], ns.b00)
    fid2 = ns.qubits.fidelity([swapSim.qC, swapSim.qB_C], ns.b00)
    print("Fidelity of A~B: ", fid1)
    print("Fidelity of B~C: ", fid2)

    w1 = (4*fid1 - 1) / 3
    w2 = (4*fid2 - 1) / 3

    wFinal = w1*w2
    fidFinal = fid1*fid2 + ((1-fid1)/3)**2 * 3

    m = swapSim.swap()
    print(m)
    print("Expected fidelity = ", fidFinal)
    # print("Fidelity A~C: ", ns.qubits.fidelity([swapSim.qA, swapSim.qC], ns.b00))
    swapSim.correct(m)
    print("Corrected A~C: ", ns.qubits.fidelity([swapSim.qA, swapSim.qC], ns.b00))

for _ in range(10):
    print("#"*15)
    simulateSwap()