import netsquid as ns
from netsquid.nodes import Node, DirectConnection
from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel
from netsquid.components import T1T2NoiseModel
from netsquid.components.qmemory import QuantumMemory
import numpy as np
import math

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

def secret_key_rate_from_fidelity(w, avg_time):
    def h(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    eX = (1-w)/2
    eZ = (1-w)/2
    r = max(0, (1 - h(eX) - h(eZ)))

    return r / avg_time


def secret_key_rate_from_density_matrix(density_matrix, avg_time):
    def h(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    e01 =  np.real((np.conj(ns.s01).T @ density_matrix @ ns.s01)[0, 0])
    e10 =  np.real((np.conj(ns.s10).T @ density_matrix @ ns.s10)[0, 0])
    
    e_h01 =  np.real((np.conj(ns.h01).T @ density_matrix @ ns.h01)[0, 0])
    e_h10 =  np.real((np.conj(ns.h10).T @ density_matrix @ ns.h10)[0, 0])

    eZ = e01 + e10
    eX = e_h01 + e_h10
    r = max(0, (1 - h(eX) - h(eZ)))

    return r / avg_time