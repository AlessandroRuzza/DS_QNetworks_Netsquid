import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import swap_then_distill as distil
from core import *

def test_distillation():
    """Test basic distillation functionality."""
    success = False
    while not success:
        ns.sim_reset()
        qA1, qC1 = bell_pair()
        qA2, qC2 = bell_pair()
        nodeA = Node("A", qmemory=make_lossy_mem(100,20,2, "memA"))
        nodeC = Node("C", qmemory=make_lossy_mem(100,20,2, "memC"))

        nodeA.qmemory.put([qA1, qA2], positions=[0,1]) #type:ignore
        nodeC.qmemory.put([qC1, qC2], positions=[0,1]) #type:ignore

        ns.sim_run(duration=1)

        distiller = distil.Distillation(nodeA, nodeC, 2)
        success = distiller.distill()
        if success:
            qA = distiller.qubitA(0)
            qC = distiller.qubitC(0)
            # print("Start fidelity: ", ns.qubits.fidelity((qA1, qC1), ns.b00))
            # print("Final fidelity: ", ns.qubits.fidelity((qA, qC), ns.b00))
            return ns.qubits.fidelity((qA, qC), ns.b00)
    return 0 # Unreachable

def run_all_tests():
    """Run all manual tests."""
    print("Running all distillation tests...\n")
    oldRes = 0
    for _ in range(int(1e5)):
        res = test_distillation()
        if res < 0.99:
            print("LESS THAN 0.99! ", res)
        if res != oldRes:
            print("New value: ", res)
            oldRes = res
    print("\nAll tests completed.")


if __name__ == "__main__":
    run_all_tests()