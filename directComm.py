import netsquid as ns
from netsquid.nodes import DirectConnection, Node
from netsquid.protocols import NodeProtocol
from netsquid.components import FibreDelayModel, FibreLossModel, QuantumChannel, Message
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

class SendProtocol(NodeProtocol):
    def __init__(self, node, stop_flag):
        super().__init__(node)
        self.qbitSent = 0
        self.stop_flag = stop_flag

    def produce_bell_pair(self):
        q1, q2 = ns.qubits.create_qubits(2)
        ns.qubits.operate([q1,], ns.H)
        ns.qubits.operate([q1,q2], ns.CNOT)
        return (q1, q2)

    def run(self):
        port = self.node.ports["qubitIO"]
        while not self.stop_flag[0]:
            self.qbitSent += 1
            qubit_id = self.qbitSent

            qubits = self.produce_bell_pair()
            msg = Message(items=[qubits[1]], meta={"id": qubit_id})
            port.tx_output(msg)
            self.qubit = qubits[0]

            # Time in nanoseconds
            yield self.await_timer(50e6)  # 50 microseconds

class ReceiveProtocol(NodeProtocol):
    def __init__(self, node, stop_flag):
        super().__init__(node)
        self.arrival_time = None
        self.stop_flag = stop_flag
        self.received_id = None
        self.fidelity = None

    def run(self):
        port = self.node.ports["qubitIO"]
        # Wait (yield) until input has arrived on our port:
        yield self.await_port_input(port)
        self.stop_flag[0] = True
        current_time = ns.sim_time(magnitude=ns.MICROSECOND)
        self.arrival_time = current_time

        # We received a qubit.
        msg = port.rx_input()
        self.qubit = msg.items[0]
        self.received_id = msg.meta["meta"]["id"]
        
def setup_sim(
    shots=100,
    distance=20,
    p_loss_init=0.0,
    p_loss_length=0.0,
    depolar_freq=10_000,
):
    """
    Setup and run a Netsquid simulation with specified parameters.
    
    Parameters:
    - distance (int): Distance between nodes in km.
    - p_loss_init (float): Initial loss probability.
    - p_loss_length (float): Loss probability per km.
    - depolar_freq (int): Depolarization frequency in Hz.

    Returns:
    - results (list of tuples): Each tuple contains
      (simulation_end_time, total_qubits_sent, arrival_time, fidelity) for each simulation
    """
    results = []
    for _ in range(shots):
        ns.sim_reset()

        nodeA, nodeB = create_directConnected_nodes(
            distance, [p_loss_init, p_loss_length], depolar_freq
        )

        stop_flag = [False]  # Mutable flag to signal stopping
        AProtocol = SendProtocol(nodeA, stop_flag)
        BProtocol = ReceiveProtocol(nodeB, stop_flag)

        AProtocol.start()
        BProtocol.start()

        stats = ns.sim_run(magnitude=ns.MICROSECOND)

        simulation_end_time = ns.sim_time(magnitude=ns.MICROSECOND)
        total_qubits_sent = BProtocol.received_id
        arrival_time = BProtocol.arrival_time

        # Calculate fidelity with respect to ideal state
        ideal_state = ns.qubits.ketstates.b00  # |00⟩ + |11> state
        q1 = AProtocol.qubit 
        q2 = BProtocol.qubit
        fidelity = ns.qubits.fidelity([q1, q2], ideal_state)

        results.append((simulation_end_time, total_qubits_sent, arrival_time, fidelity))

    return results

if __name__ == "__main__":
    results = setup_sim(p_loss_init=0.5, shots=int(1e4))
    _, nTries, _, fidelities = zip(*results)
    avgTries = float(sum(nTries)) / len(nTries)
    print(f"Average tries = {avgTries}")

    unique_fids = set(fidelities)
    print("Num diff. fidelities =", len(unique_fids), f" ; {unique_fids}")