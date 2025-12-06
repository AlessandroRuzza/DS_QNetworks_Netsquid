import netsquid as ns
from netsquid.nodes import Node, DirectConnection
from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel
from netsquid.components import T1T2NoiseModel
from netsquid.components.qmemory import QuantumMemory

class SymmetricConnection(DirectConnection):
    def __init__(self, name, L:int, loss_model, delay_model, noise_model) -> None:
        self.modelDict = {
                    'quantum_noise_model': noise_model,
                    'quantum_loss_model': loss_model,
                    'delay_model': delay_model
                    }
        self.length = L
        abChannel = QuantumChannel("A->B", length=L, models=self.modelDict)
        baChannel = QuantumChannel("B->A", length=L, models=self.modelDict)
        super().__init__(name, abChannel, baChannel)

    def copy(self, newName):
        return SymmetricConnection(
            name=newName,
            L=self.length,
            loss_model=self.modelDict['quantum_loss_model'],
            delay_model=self.modelDict['delay_model'],
            noise_model=self.modelDict['quantum_noise_model']
        )

def bell_pair():
    q1, q2 = ns.qubits.create_qubits(2)
    ns.qubits.operate([q1], ns.H)
    ns.qubits.operate([q1, q2], ns.CNOT)
    return q1, q2

def make_lossy_mem(T1_mem, T2_mem, n=2, name="mem"):
    mem = QuantumMemory(name, num_positions=n)
    if T1_mem > 0 or T2_mem > 0:
        mem_noise = T1T2NoiseModel(T1=T1_mem, T2=T2_mem) # type:ignore
        for mem_pos in mem.mem_positions:
            mem_pos.models["noise_model"] = mem_noise

    return mem
