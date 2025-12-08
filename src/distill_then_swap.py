from core import *
import netsquid as ns
from netsquid.components import FibreDelayModel, FibreLossModel, T1T2NoiseModel
import netsquid.components.instructions as instr
import longRange
from scenarios import travel_ns_km

def create_repeater_nodes(
    distance: int,
    p_loss_init: float,
    p_loss_length: float,
    t1_channel: float,
    t2_channel: float,
    T1_mem: float,
    T2_mem: float,
    num_connections:int=1,
):
    portsA = [f"port{i}AB" for i in range(num_connections)]
    portsC = [f"port{i}BC" for i in range(num_connections)]
    portsB = [*portsA, *portsC]

    nodeA = Node("nodeA", port_names=portsA, qmemory=make_lossy_mem(0, 0, num_connections, "memA"))
    nodeB = Node("nodeB", port_names=portsB, qmemory=make_lossy_mem(T1_mem, T2_mem, 2*num_connections, "memB"))
    nodeC = Node("nodeC", port_names=portsC, qmemory=make_lossy_mem(0, 0, num_connections, "memC"))

    loss_model = FibreLossModel(p_loss_init, p_loss_length)
    delay_model = FibreDelayModel()
    noise_model = T1T2NoiseModel(t1_channel, t2_channel) # type:ignore

    conn = SymmetricConnection("channel", distance, loss_model, delay_model, noise_model)
    return nodeA, nodeB, nodeC, conn

def add_connection(nodes: list[Node], portName:str, connectionTemplate: SymmetricConnection, connectionPostFix:str):
    l = [chr(ord('A') + i) for i in range(len(nodes))]
    for i in range(len(nodes) - 1):
        node1 = nodes[i]
        node2 = nodes[i + 1]
        conn = connectionTemplate.copy(f"{l[i]}{l[i+1]}_{connectionPostFix}")
        
        node1.connect_to(node2, connection=conn, local_port_name=f"{portName}{l[i]}{l[i+1]}", remote_port_name=f"{portName}{l[i]}{l[i+1]}")

class Distillation():
    class Failure(Exception):
        pass

    def __init__(self, nodeA: Node, nodeB: Node, num_links: int, isDistillAB:bool):
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.num_links = num_links
        self.isDistillAB = isDistillAB

    @property
    def memA(self) -> QuantumMemory:
        return self.nodeA.qmemory
    @property
    def memB(self) -> QuantumMemory:
        return self.nodeB.qmemory

    from netsquid.examples.purify import Distil as Distil
    INSTR_ROT_A = Distil._INSTR_Rx
    INSTR_ROT_B = Distil._INSTR_RxC

    def _distill_operation(self, idx1_A, idx2_A):
        idx1_B = 2*idx1_A if self.isDistillAB else 2*idx1_A+1
        idx2_B = 2*idx2_A if self.isDistillAB else 2*idx2_A+1
        
        self.INSTR_ROT_A(self.memA, [idx1_A])
        self.INSTR_ROT_A(self.memA, [idx2_A])

        self.INSTR_ROT_B(self.memB, [idx1_B])
        self.INSTR_ROT_B(self.memB, [idx2_B])
        
        instr.INSTR_CNOT(self.memA, [idx1_A, idx2_A])
        instr.INSTR_CNOT(self.memB, [idx1_B, idx2_B])
        
        mA = instr.INSTR_MEASURE(self.memA, [idx2_A])[0] #type: ignore
        mB = instr.INSTR_MEASURE(self.memB, [idx2_B])[0] #type: ignore
        return mA == mB

    def _compactMem(self, mem, results, isMemB:bool):
        nextFreeSpot = 0
        offset = 1 if not self.isDistillAB and isMemB else 0
        for i, res in enumerate(results):
            if res:
                idx = 4*i if isMemB else 2*i
                idx += offset
                q = mem.pop(idx)[0]
                mem.put([q], positions=[nextFreeSpot + offset])
                nextFreeSpot += 2

    def _distillPass(self):
        results = []
        for i in range(0, self.num_links-1, 2):
            results.append(self._distill_operation(i, i+1))
        # Compact memory
        self._compactMem(self.memA, results, isMemB=False)
        self._compactMem(self.memB, results, isMemB=True)

        n_successes = results.count(True)
        self.num_links = n_successes + self.num_links % 2

    def distill(self):
        while self.num_links > 1:
            self._distillPass()
        return self.num_links == 1

def reset_state(nodeA, nodeC, num_links:int):
    state = []
    for i in range(num_links):
        state.append({
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
        })
    return state

class RepeaterNoSwapProtocol(longRange.RepeaterProtocol):
    def run(self):
        while not self.state["have_AB"] or not self.state["have_BC"]:
            expr = yield self.await_port_input(self.portAB) | self.await_port_input(self.portBC)      

            if expr.first_term.value:
                self._handle_recv_AB()
            if expr.second_term.value:
                self._handle_recv_BC()

        assert self.state["have_AB"] and self.state["have_BC"]

def setup_distill_then_swap_sim(
    shots: int,
    distance: int,
    p_loss_init: float = 0.0,
    p_loss_length: float = 0.0,
    T1_channel: float = 0.0,
    T2_channel: float = 0.0,
    T1_mem: float = 0.0,
    T2_mem: float = 0.0,
):
    # --- Initialize results and constants ---
    results = []
    C = 2e5 / 1e9  # km/ns
    num_links = 2 # n parallel entangled per link
    timeout = 2 * distance / C  # waiting time for generation

    # main loop
    for _ in range(shots):
        success = False
        attempts_total = 0
        F_AC = -1

        # generation loop
        while not success:
            ns.sim_reset() # fail

            # create network
            nodeA, nodeB, nodeC, connection = create_repeater_nodes(
                distance, p_loss_init, p_loss_length,
                T1_channel, T2_channel,
                T1_mem, T2_mem,
                num_connections=2,
            )

            state = reset_state(nodeA, nodeC, num_links)
            protoA = []
            protoB = []
            protoC = []

            # Setup connections
            for i in range(num_links):
                add_connection([nodeA, nodeB, nodeC], f"port{i}", connection, f"channel{i}")
                protoA.append(longRange.SendShortLink(nodeA, f"port{i}AB", "AB", state[i], timeout, n_link=i))
                protoC.append(longRange.SendShortLink(nodeC, f"port{i}BC", "BC", state[i], timeout, n_link=i))
                protoB.append(RepeaterNoSwapProtocol(nodeB, state[i], n_link=i))

                protoA[i].start()
                protoB[i].start()
                protoC[i].start()

            ns.sim_run()

            # distillation on A-B and B-C links
            distillerAB = Distillation(nodeA, nodeB, num_links, isDistillAB=True)
            distillerBC = Distillation(nodeC, nodeB, num_links, isDistillAB=False)
            successAB = distillerAB.distill()
            successBC = distillerBC.distill()

            # Count attempts for link generation
            attempts_AB = max([s["id_AB"] for s in state])
            attempts_BC = max([s["id_BC"] for s in state])
            attempts_total += max(attempts_AB, attempts_BC)

            # --- Retry if distillation fails ---
            if not (successAB and successBC):
                continue  # try again

            # --- Trigger swap at B ---
            protoB[0]._swap()

            # Measure A~C fidelity
            try:
                qA = nodeA.qmemory.peek(0)[0]
                qC = nodeC.qmemory.peek(0)[0]
                F_AC = ns.qubits.fidelity((qA, qC), ns.b00, squared=True)
            except Exception:
                continue # retry

            success = True # done

        # Store
        assert F_AC >= 0
        sim_end_time = ns.sim_time(magnitude=ns.MICROSECOND)
        results.append(
            (sim_end_time, attempts_total, F_AC)
        )

    return results

if __name__ == "__main__":
    t1 = travel_ns_km*100
    t2 = travel_ns_km*20
    results = setup_distill_then_swap_sim(100, 10, 
                                T1_channel=t1, 
                                T2_channel=t2,
                                T1_mem=t1, 
                                T2_mem=t2, 
                                )

    _, attempts_total, fidelities = zip(*results)

    avg_fidelity = sum(fidelities) / len(fidelities)
    avg_attempts = sum(attempts_total) / len(attempts_total)

    print("Avg fidelity = ", avg_fidelity)
    print("Avg time units = ", avg_attempts)
    