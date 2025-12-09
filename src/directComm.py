import netsquid as ns
from netsquid.nodes import Node
from netsquid.protocols import NodeProtocol
from netsquid.components import FibreDelayModel, FibreLossModel, Message  #type:ignore
from netsquid.components import T1T2NoiseModel
from scenarios import args
ns.set_qstate_formalism(ns.qubits.DenseDMRepr)

from core import *

def create_directConnected_nodes(distance: int, p: list[float], t1:float,t2:float):
    assert len(p) >= 2
    portName = "qubitIO"
    t1_A, t2_A = (t1, t2) if args.noisyACMemories else (0,0)
    nodeA = Node("nodeA", port_names=[portName], qmemory=make_lossy_mem(t1_A,t2_A,1))
    nodeB = Node("nodeB", port_names=[portName])
    conn = SymmetricConnection("AB_channel", distance, 
                                     FibreLossModel(p[0], p[1]), FibreDelayModel(), 
                                     T1T2NoiseModel(t1,t2)) #type: ignore
    nodeA.connect_to(remote_node=nodeB, connection=conn,
                        local_port_name=portName, remote_port_name=portName)
    return nodeA, nodeB

class SendProtocol(NodeProtocol):
    def __init__(self, node, stop_flag, timeout_us:float):
        """
        timeout_us (float) : Retransmit timeout expressed in nanoseconds.
        """
        super().__init__(node)
        self.qbitSent = 0
        self.stop_flag = stop_flag
        self.timeout = timeout_us

    @property
    def qubit(self):
        return self.node.qmemory.peek(0)[0]

    def run(self):
        port = self.node.ports["qubitIO"]
        while not self.stop_flag[0]:
            self.qbitSent += 1
            qubit_id = self.qbitSent

            qubits = bell_pair()
            msg = Message(items=[qubits[1]], meta={"id": qubit_id})
            port.tx_output(msg)
            self.node.qmemory.put(qubits[0], positions=0)

            # Time in nanoseconds
            yield self.await_timer(self.timeout)  # time in nanoseconds should be roughly 1 round trip

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
    shots,
    distance,
    p_loss_init=0.0,
    p_loss_length=0.0,
    t1:float=0.0,
    t2:float=0.0,
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
    C = 2e5 / 1e9 # [km/ns] speed of light in fibre
    for _ in range(shots):
        ns.sim_reset()

        nodeA, nodeB = create_directConnected_nodes(
            distance, [p_loss_init, p_loss_length], t1, t2
        )

        stop_flag = [False]  # Mutable flag to signal stopping
        AProtocol = SendProtocol(nodeA, stop_flag, 2*distance/C) # Timeout = 1 round trip = 2*dist / C [ns]
        BProtocol = ReceiveProtocol(nodeB, stop_flag)

        AProtocol.start()
        BProtocol.start()

        stats = ns.sim_run(magnitude=ns.MICROSECOND) 

        simulation_end_time = ns.sim_time(magnitude=ns.MICROSECOND)
        total_qubits_sent = BProtocol.received_id
        arrival_time = BProtocol.arrival_time

        # Calculate fidelity with respect to ideal state
        ideal_state = ns.b00  # |00⟩ + |11> state
        q1 = AProtocol.qubit 
        q2 = BProtocol.qubit
        fidelity = ns.qubits.qubitapi.fidelity([q1, q2], ideal_state, squared=True)
        keyRate = secret_key_rate_from_density_matrix(
                            ns.qubits.reduced_dm([q1,q2]), 
                            ns.sim_time(magnitude=ns.SECOND)
                            )

        results.append((simulation_end_time, total_qubits_sent, arrival_time, fidelity, keyRate))

    return results

if __name__ == "__main__":
    import sys
    results = setup_sim(distance=int(sys.argv[1]), p_loss_init=0.5, shots=int(50), t1=0, t2=0)
    _, nTries, arrival_times, fidelities = zip(*results)
    avgTries = float(sum(nTries)) / len(nTries)
    print(f"Average tries = {avgTries}")
    avgTime = float(sum(arrival_times)) / len(arrival_times)
    print(f"Average arrival time = {avgTime}")

    unique_fids = set(fidelities)
    print("Num diff. fidelities =", len(unique_fids), f" ; {unique_fids}")