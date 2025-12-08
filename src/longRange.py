import netsquid as ns
from netsquid.nodes import Node, DirectConnection
from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel, Message #type:ignore
from netsquid.components import FibreDelayModel, FibreLossModel, T1T2NoiseModel
from netsquid.components.qmemory import QuantumMemory
import netsquid.components.instructions as instr

from core import *

ns.set_qstate_formalism(ns.qubits.DenseDMRepr)

def create_repeater_nodes(
    distance: int,
    p_loss_init: float,
    p_loss_length: float,
    T1_channel: float,
    T2_channel: float,
    T1_mem: float,
    T2_mem: float,
):
    portA = "port0AB"
    portB_AB = "port0AB"
    portB_BC = "port0BC"
    portC = "port0BC"

    nodeA = Node("nodeA", port_names=[portA], qmemory=make_lossy_mem(T1_mem, T2_mem, 1, "memA"))
    nodeB = Node("nodeB", port_names=[portB_AB, portB_BC], qmemory=make_lossy_mem(T1_mem, T2_mem, 2, "memB"))
    nodeC = Node("nodeC", port_names=[portC], qmemory=make_lossy_mem(T1_mem, T2_mem, 1, "memC"))

    loss_model = FibreLossModel(p_loss_init, p_loss_length)
    delay_model = FibreDelayModel()
    noise_model = T1T2NoiseModel(T1_channel, T2_channel) # type:ignore

    conn_AB = SymmetricConnection("AB_channel", distance, loss_model, delay_model, noise_model)
    conn_BC = SymmetricConnection("BC_channel", distance, loss_model, delay_model, noise_model)

    nodeA.connect_to(nodeB, connection=conn_AB,
                     local_port_name=portA, remote_port_name=portB_AB)
    nodeB.connect_to(nodeC, connection=conn_BC,
                     local_port_name=portB_BC, remote_port_name=portC)

    return nodeA, nodeB, nodeC, conn_AB


class SendShortLink(NodeProtocol):
    # part 1

    def __init__(self, node:Node, port_name: str, link_label: str,
                 state: dict, timeout_ns: float, n_link:int=0):
        super().__init__(node)
        self.port_name = port_name
        self.link_label = link_label
        self.state = state
        self.timeout = timeout_ns
        self.attempt_id = 0
        self.n_link = n_link

    @property
    def qmem(self):
        return self.node.qmemory
    
    def stop_flag(self) -> bool:
        if self.link_label == "AB":
            return self.state["have_AB"]
        else:
            return self.state["have_BC"]

    def run(self):
        port = self.node.ports[self.port_name]
        while not self.stop_flag() and not self.state["done"]:
            self.attempt_id += 1
            q1, q2 = bell_pair()
            self.qmem.put(q1, positions=[self.n_link])

            msg = Message(items=[q2], meta={"link": self.link_label,
                                            "id": self.attempt_id})
            port.tx_output(msg)

            yield self.await_timer(self.timeout)

class RepeaterProtocol(NodeProtocol):
    # part 2

    def __init__(self, node: QuantumMemory,
                 state: dict,
                 n_link:int = 0):
        super().__init__(node)
        self.state = state
        self.n_link = n_link

    @property
    def memB(self):
        return self.node.qmemory
    
    @property
    def portAB(self):
        return self.node.ports[f"port{self.n_link}AB"]
    @property
    def portBC(self):
        return self.node.ports[f"port{self.n_link}BC"]

    def _handle_recv_AB(self):
        msg = self.portAB.rx_input()
        meta = msg.meta["meta"]
        q_from_A = msg.items[0]
        pair_id = meta["id"]

        self.memB.put(q_from_A, positions=2*self.n_link)
        self.state["id_AB"] = pair_id
        self.state["have_AB"] = True

    def _handle_recv_BC(self):
        msg = self.portBC.rx_input()
        meta = msg.meta["meta"]
        q_from_C = msg.items[0]
        pair_id = meta["id"]

        self.memB.put(q_from_C, positions=2*self.n_link+1)
        self.state["id_BC"] = pair_id
        self.state["have_BC"] = True

    def _correct(self, m, mem):
        if m==1: instr.INSTR_X(mem, [self.n_link])
        if m==2: instr.INSTR_Y(mem, [self.n_link]) # NOTE: bell measure = 2 means A~C == b11
        if m==3: instr.INSTR_Z(mem, [self.n_link]) # NOTE: bell measure = 3 means A~C == b10
    
    def _swap(self):
        m = instr.INSTR_MEASURE_BELL(self.memB, [2*self.n_link, 2*self.n_link+1])[0] #type: ignore
        self._correct(m, self.state["C_mem"])
        
        self.state["qA"] = self.state["A_mem"].peek(self.n_link)[0]
        self.state["qC"] = self.state["C_mem"].peek(self.n_link)[0]
        
        F_AC = ns.qubits.fidelity([self.state["qA"], self.state["qC"]], ns.b00, squared=True)

        self.state["F_AC"] = F_AC
        self.state["m"] = m
        self.state["swap_time"] = ns.sim_time(magnitude=ns.MICROSECOND)
        self.state["success"] = True
        self.state["done"] = True
    
    def run(self):
        while not self.state["done"]:
            while not self.state["have_AB"] or not self.state["have_BC"]:
                expr = yield self.await_port_input(self.portAB) | self.await_port_input(self.portBC)      

                if expr.first_term.value:
                    self._handle_recv_AB()
                if expr.second_term.value:
                    self._handle_recv_BC()

            assert self.state["have_AB"] and self.state["have_BC"]
            self._swap()

def setup_longrange_sim(
    shots: int,
    distance: int,
    p_loss_init: float = 0.0,
    p_loss_length: float = 0.0,
    t1_channel: float = 0.0,
    t2_channel: float = 0.0,
    T1_mem: float = 0.0,
    T2_mem: float = 0.0,
):
    results = []
    C = 2e5 / 1e9  # km/ns

    for _ in range(shots):
        ns.sim_reset()

        nodeA, nodeB, nodeC, _ = create_repeater_nodes(
            distance, p_loss_init, p_loss_length,
            t1_channel, t2_channel,
            T1_mem, T2_mem,
        )

        state = {
            "done": False,
            "have_AB": False,
            "have_BC": False,
            "A_mem": nodeA.qmemory,
            "C_mem": nodeC.qmemory,
            "id_AB": None,
            "id_BC": None,
            "qA": None,
            "qC": None,
            "F_AC": None,
            "swap_time": None,
            "success": False,
            "m":None,
        }

        timeout = 2 * distance / C  # ns

        protoA = SendShortLink(nodeA, "port0AB", "AB", state, timeout)
        protoC = SendShortLink(nodeC, "port0BC", "BC", state, timeout)
        protoB = RepeaterProtocol(nodeB, state)

        protoA.start()
        protoB.start()
        protoC.start()

        ns.sim_run(magnitude=ns.MICROSECOND)

        sim_end_time = ns.sim_time(magnitude=ns.MICROSECOND)
        attempts_AB = state["id_AB"]
        attempts_BC = state["id_BC"]
        attempts_total = max(attempts_AB, attempts_BC) if attempts_AB and attempts_BC else None
        swap_time = state["swap_time"]
        F_AC = state["F_AC"]

        results.append(
            (sim_end_time, attempts_AB, attempts_BC, attempts_total, swap_time, F_AC)
        )

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python src/longRange.py <distance_km> <shots>")
        sys.exit(1)

    distance = int(sys.argv[1])
    shots = int(sys.argv[2])
    
    travel_time = distance / FibreDelayModel.c #type: ignore
    results = setup_longrange_sim(
        shots=shots,
        distance=distance,
        p_loss_init=0.5,
        p_loss_length=0.2,
        t1_channel=0.0,
        t2_channel=0.0,
        T1_mem = travel_time/2,
        T2_mem = travel_time/8,
    )

    _, attempts_AB, attempts_BC, attempts_total, swap_times, fidelities = zip(*results)

    avg_attempts_AB = sum(a for a in attempts_AB if a is not None) / len(attempts_AB)
    avg_attempts_BC = sum(a for a in attempts_BC if a is not None) / len(attempts_BC)
    avg_attempts_total = sum(a for a in attempts_total if a is not None) / len(attempts_total)
    avg_swap_time = sum(t for t in swap_times if t is not None) / len(swap_times)
    avg_fidelity = sum(f for f in fidelities if f is not None) / len(fidelities)

    print(f"Distance           = {distance} km")
    print(f"Shots              = {shots}")
    print(f"Avg attempts A-B   = {avg_attempts_AB:.2f}")
    print(f"Avg attempts B-C   = {avg_attempts_BC:.2f}")
    print(f"Avg attempts total = {avg_attempts_total:.2f}")
    print(f"Avg swap time      = {avg_swap_time:.3f} microseconds")
    print(f"Avg fidelity A~C   = {avg_fidelity:.4f}")
