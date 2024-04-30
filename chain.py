#This program aims to reproduce the one d chain with netsquid
#Let us first import some stuff
import netsquid as ns
from netsquid.nodes import Node, Network
from netsquid.protocols import Protocol, NodeProtocol
from netsquid.components import QuantumChannel, QuantumProgram, ClassicalChannel
from netsquid.qubits import qubitapi as qapi
from netsquid.util.datacollector import DataCollector

#defining useful things
def create_linear_network(num_nodes):
    nodes = [Node(f'Node_{i}') for i in range(num_nodes)]
    network = Network("Linear Quantum Network")
    for node in nodes:
        network.add_node(node)
    return network, nodes
num_nodes=5
def connect_nodes_with_channels(network, nodes, qchannel_length, cchannel_length):
    for i in range(len(nodes) - 1):
        qc = QuantumChannel(f"QChannel_{i}_{i+1}", length=qchannel_length, models=None)
        cc = ClassicalChannel(f"CChannel_{i}_{i+1}", length=cchannel_length)
        network.add_connection(nodes[i], nodes[i + 1], channel_to=qc, label='quantum')
        network.add_connection(nodes[i], nodes[i + 1], channel_to=cc, label='classical')

class EntanglementDistributionProtocol(NodeProtocol):
    def __init__(self, node, remote_node, channel):
        super().__init__(node)
        self.remote_node = remote_node
        self.channel = channel

    def run(self):
        # Create entanglement
        q1, q2 = qapi.create_qubits(2)
        qapi.operate(q1, ns.H)  # Apply Hadamard gate to the first qubit
        qapi.operate([q1, q2], ns.CNOT)  # Apply CNOT gate with q1 as control and q2 as target
        yield self.channel.send(self.remote_node, q2)
        self.send_signal(signal_label="entanglement_success", qubit=q1)
    

network, nodes = create_linear_network(num_nodes)
connect_nodes_with_channels(network, nodes, 1e-3, 1e-3)  # Assuming very short channel lengths for simplicity

protocols = []
for i in range(len(nodes) - 1):
    protocol = EntanglementDistributionProtocol(nodes[i], nodes[i + 1], network.get_connection(nodes[i], nodes[i + 1], label='quantum'))
    protocols.append(protocol)
    protocol.start()

ns.sim_run()
