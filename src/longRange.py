import netsquid as ns
from netsquid.nodes import Node, DirectConnection
from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel, Message
from netsquid.components import FibreDelayModel, FibreLossModel, T1T2NoiseModel
from netsquid.components.qmemory import QuantumMemory
import netsquid.components.instructions as instr

ns.set_qstate_formalism(ns.qubits.DenseDMRepr)

class SymmetricConnection(DirectConnection):
    def __init__(self, name, L: int, loss_model, delay_model, noise_model) -> None:
        model_dict = {
            "quantum_noise_model": noise_model,
            "quantum_loss_model": loss_model,
            "delay_model": delay_model,
        }
        ch_atob = QuantumChannel(f"{name}_AtoB", length=L, models=model_dict)
        ch_btoa = QuantumChannel(f"{name}_BtoA", length=L, models=model_dict)
        super().__init__(name, ch_atob, ch_btoa)


def bell_pair():
    q1, q2 = ns.qubits.create_qubits(2)
    ns.qubits.operate([q1], ns.H)
    ns.qubits.operate([q1, q2], ns.CNOT)
    return q1, q2

def make_lossy_mem(T1_mem, T2_mem, n=2):
    mem = QuantumMemory("memB", num_positions=n)
    mem_noise = T1T2NoiseModel(T1=T1_mem, T2=T2_mem) # type:ignore
    for mem_pos in mem.mem_positions:
        mem_pos.models["noise_model"] = mem_noise

    return mem

def create_repeater_nodes(
    distance: int,
    p_loss_init: float,
    p_loss_length: float,
    t1_channel: float,
    t2_channel: float,
    T1_mem: float,
    T2_mem: float,
):
    portA = "portAB"
    portB_AB = "portAB"
    portB_BC = "portBC"
    portC = "portBC"

    nodeA = Node("nodeA", port_names=[portA], qmemory=make_lossy_mem(T1_mem, T2_mem, 1))
    nodeB = Node("nodeB", port_names=[portB_AB, portB_BC], qmemory=make_lossy_mem(T1_mem, T2_mem, 2))
    nodeC = Node("nodeC", port_names=[portC], qmemory=make_lossy_mem(T1_mem, T2_mem, 1))

    loss_model = FibreLossModel(p_loss_init, p_loss_length)
    delay_model = FibreDelayModel()
    noise_model = T1T2NoiseModel(t1_channel, t2_channel) # type:ignore

    conn_AB = SymmetricConnection("AB_channel", distance, loss_model, delay_model, noise_model)
    conn_BC = SymmetricConnection("BC_channel", distance, loss_model, delay_model, noise_model)

    nodeA.connect_to(nodeB, connection=conn_AB,
                     local_port_name=portA, remote_port_name=portB_AB)
    nodeB.connect_to(nodeC, connection=conn_BC,
                     local_port_name=portB_BC, remote_port_name=portC)

    return nodeA, nodeB, nodeC


class SendShortLink(NodeProtocol):
    # part 1

    def __init__(self, node:Node, port_name: str, link_label: str,
                 stop_flag: list[bool], state: dict, timeout_ns: float):
        super().__init__(node)
        self.qmem = node.qmemory
        self.port_name = port_name
        self.link_label = link_label
        self.stop_flag = stop_flag
        self.state = state
        self.timeout = timeout_ns
        self.attempt_id = 0

    def run(self):
        port = self.node.ports[self.port_name]
        while not self.stop_flag[0] and not self.state["done"]:
            self.attempt_id += 1
            q1, q2 = bell_pair()
            self.qmem.put(q1, positions=[0])
            if self.link_label == "AB":
                self.state["A_qubits"][self.attempt_id] = self.qmem
            else:
                self.state["C_qubits"][self.attempt_id] = self.qmem

            msg = Message(items=[q2], meta={"link": self.link_label,
                                            "id": self.attempt_id})
            port.tx_output(msg)

            yield self.await_timer(self.timeout)


class RepeaterProtocol(NodeProtocol):
    # part 2

    def __init__(self, node: QuantumMemory,
                 stop_flag_AB: list[bool], stop_flag_BC: list[bool],
                 state: dict):
        super().__init__(node)
        self.memB = node.qmemory
        self.stop_flag_AB = stop_flag_AB
        self.stop_flag_BC = stop_flag_BC
        self.state = state

    def run(self):
        portAB = self.node.ports["portAB"]
        portBC = self.node.ports["portBC"]

        while not self.state["done"]:
            if not self.state["have_AB"]:
                yield self.await_port_input(portAB)
                msg = portAB.rx_input()
                meta = msg.meta["meta"]
                q_from_A = msg.items[0]
                pair_id = meta["id"]

                self.memB.put(q_from_A, positions=0)
                self.state["id_AB"] = pair_id
                self.state["have_AB"] = True
                self.stop_flag_AB[0] = True

            if not self.state["have_BC"]:
                yield self.await_port_input(portBC)
                msg = portBC.rx_input()
                meta = msg.meta["meta"]
                q_from_C = msg.items[0]
                pair_id = meta["id"]

                self.memB.put(q_from_C, positions=1)
                self.state["id_BC"] = pair_id
                self.state["have_BC"] = True
                self.stop_flag_BC[0] = True

            if self.state["have_AB"] and self.state["have_BC"]:
                self.state["qA"] = self.state["A_qubits"][self.state["id_AB"]].peek(0)[0]
                self.state["qC"] = self.state["C_qubits"][self.state["id_BC"]].peek(0)[0]
                m = instr.INSTR_MEASURE_BELL(self.memB, [0,1])[0] #type: ignore

                f00 = ns.qubits.fidelity([self.state["qA"], self.state["qC"]], ns.b00)
                f01 = ns.qubits.fidelity([self.state["qA"], self.state["qC"]], ns.b01)
                f10 = ns.qubits.fidelity([self.state["qA"], self.state["qC"]], ns.b10)
                f11 = ns.qubits.fidelity([self.state["qA"], self.state["qC"]], ns.b11)
                F_AC = max(f00, f01, f10, f11)

                self.state["F_AC"] = F_AC
                self.state["m"] = m
                self.state["swap_time"] = ns.sim_time(magnitude=ns.MICROSECOND)
                self.state["success"] = True
                self.state["done"] = True


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

        nodeA, nodeB, nodeC = create_repeater_nodes(
            distance, p_loss_init, p_loss_length,
            t1_channel, t2_channel,
            T1_mem, T2_mem,
        )

        state = {
            "done": False,
            "have_AB": False,
            "have_BC": False,
            "A_qubits": {},
            "C_qubits": {},
            "id_AB": None,
            "id_BC": None,
            "qA": None,
            "qC": None,
            "F_AC": None,
            "swap_time": None,
            "success": False,
            "m":None,
        }

        stop_AB = [False]
        stop_BC = [False]
        timeout = 2 * distance / C  # ns

        protoA = SendShortLink(nodeA, "portAB", "AB", stop_AB, state, timeout)
        protoC = SendShortLink(nodeC, "portBC", "BC", stop_BC, state, timeout)
        protoB = RepeaterProtocol(nodeB, stop_AB, stop_BC, state)

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
