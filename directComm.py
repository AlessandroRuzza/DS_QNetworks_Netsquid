import netsquid as ns
from netsquid.nodes import DirectConnection, Node
from netsquid.components import FibreDelayModel, FibreLossModel, QuantumChannel
from netsquid.components import DepolarNoiseModel, DephaseNoiseModel

class SymmetricDirectConnection(DirectConnection):
    def __init__(self, name, L:int, loss_model, delay_model, noise_model) -> None:
        modelDict = {
                    'quantum_noise_model': noise_model,
                    'quantum_loss_model': loss_model,
                    'delay_model': delay_model
                    }
        abChannel = QuantumChannel("A->B", length=L, models=modelDict)
        baChannel = QuantumChannel("B->A", length=L, models=modelDict)
        super().__init__(name, abChannel, baChannel)

def create_directConnected_nodes(distance: int, p: list[float], depolar_freq):
    assert len(p) >= 2
    portName = "qubitIO"
    nodeA = Node("nodeA", port_names=[portName])
    nodeB = Node("nodeB", port_names=[portName])
    conn = SymmetricDirectConnection("AB_channel", distance, 
                                     FibreLossModel(p[0], p[1]), FibreDelayModel(), 
                                     DepolarNoiseModel(depolar_freq))
    nodeA.connect_to(remote_node=nodeB, connection=conn,
                        local_port_name=portName, remote_port_name=portName)
    return nodeA, nodeB

def start_pingPong(nodeA, nodeB):
    from pingPong import PingPongProtocol
    q = ns.qubits.create_qubits(1)
    ping = PingPongProtocol(nodeA, ns.Z, q[0])
    pong = PingPongProtocol(nodeB, ns.X)

    ping.start()
    pong.start()

def main():
    NODE_DIST = 10
    depolar_freq = 5e4
    nodeA, nodeB = create_directConnected_nodes(NODE_DIST, [0.0, 0.0], depolar_freq)   
    start_pingPong(nodeA, nodeB) 
    stats:ns.util.SimStats = ns.sim_run(duration=205, magnitude=ns.MICROSECOND) #type: ignore
    print(stats.summary())
    print("Raw data dictionary entries:")
    print(stats.data)

if __name__ == "__main__":
    main()