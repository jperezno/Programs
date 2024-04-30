import netsquid as ns
import numpy as np
from netsquid.qubits.qformalism import QFormalism


# Change to density matrix formalism:
ns.set_qstate_formalism(QFormalism.DM)
a1, a2, b1 = ns.qubits.create_qubits(3)

# put a1 into the chosen target state
ns.qubits.operate(a1, ns.H) # apply Hadamard gate to a1: |0> -> |+>

ns.qubits.operate(a1, ns.S) # apply S gate to a1: |+> -> |0_y>

print(ns.qubits.reduced_dm([a1]))

# transform a2 and b1 to the Bell state |b00> = (|00> + |11>)/sqrt(2)
ns.qubits.operate(a2, ns.H)  # apply Hadamard gate to a2
ns.qubits.operate([a2, b1], ns.CNOT)  # CNOT: a2 = control, b1 = target
print(ns.qubits.reduced_dm([a2, b1]))
