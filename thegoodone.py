# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# File: repeater_chain.py
# 
# This file is part of the NetSquid package (https://netsquid.org).
# It is subject to the NetSquid Software End User License Conditions.
# A copy of these conditions can be found in the LICENSE.md file of this package.
# 
# NetSquid Authors
# ================
# 
# NetSquid is being developed within [Quantum Internet division](https://qutech.nl/research-engineering/quantum-internet/) at QuTech.
# QuTech is a collaboration between TNO and the TUDelft.
# 
# Active authors (alphabetical):
# 
# - Tim Coopmans (scientific contributor)
# - Chris Elenbaas (software developer)
# - David Elkouss (scientific supervisor)
# - Rob Knegjens (tech lead, software architect)
# - Iñaki Martin Soroa (software developer)
# - Julio de Oliveira Filho (software architect)
# - Ariana Torres Knoop (HPC contributor)
# - Stephanie Wehner (scientific supervisor)
# 
# Past authors (alphabetical):
# 
# - Axel Dahlberg (scientific contributor)
# - Damian Podareanu (HPC contributor)
# - Walter de Jong (HPC contributor)
# - Loek Nijsten (software developer)
# - Martijn Papendrecht (software developer)
# - Filip Rozpedek (scientific contributor)
# - Matt Skrzypczyk (software contributor)
# - Leon Wubben (software developer)
# 
# The simulation engine of NetSquid depends on the pyDynAA package,
# which is developed at TNO by Julio de Oliveira Filho, Rob Knegjens, Coen van Leeuwen, and Joost Adriaanse.
# 
# Ariana Torres Knoop, Walter de Jong and Damian Podareanu from SURFsara have contributed towards the optimization and parallelization of NetSquid.
# 
# Hana Jirovska and Chris Elenbaas have built Python packages for MacOS.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

In this example we show how a simple quantum repeater chain network can be setup and simulated in NetSquid.
The module file used in this example can be located as follows:

>>> import netsquid as ns
>>> print("This example module is located at: {}".format(
...       ns.examples.repeater_chain.__file__))
This example module is located at: .../netsquid/examples/repeater_chain.py

In the `repeater example <netsquid.examples.repeater.html>`_ we simulated a single quantum repeater on a network topology consisting of two nodes
connected via a single repeater node (see for instance `[Briegel et al.] <https://arxiv.org/abs/quant-ph/9803056>`_ for more background).
To simulate a repeater chain we will extend this network topology to be a line of *N* nodes as shown below:

.. aafig::
    :textual:
    :proportional:

    +-----------+   +----------+         +------------+   +----------+
    |           |   |          |         |            |   |          |
    | "Node 0"  O---O "Node 1" O-- ooo --O "Node N-1" O---O "Node N" |
    | "(Alice)" |   |          |         |            |   | "(Bob)"  |
    |           |   |          |         |            |   |          |
    +-----------+   +----------+         +------------+   +----------+

We will refer to the outer nodes as *end nodes*, and sometimes also as Alice and Bob for convenience,
and the in between nodes as the *repeater nodes*.
The lines between the nodes represent both an entangling connection and a classical connection,
as introduced in the `teleportation example <netsquid.examples.teleportation.html>`_.
The repeaters will use a so-called `entanglement swapping scheme <https://en.wikipedia.org/wiki/Quantum_teleportation>`_ to entangle the end nodes,
which consists of the following steps:

1. generating entanglement with both of its neighbours,
2. measuring its two locally stored qubits in the Bell basis,
3. sending its own measurement outcomes to its right neighbour, and also forwarding on outcomes received from its left neighbour in this way.

Let us create the repeater chain network.
We need to create the *N* nodes, each with a quantum processor, and every pair of nodes
in the chain must be linked using an entangling connection and a classical connection.
In each entangling connection an entangled qubit generating source is available.
A schematic illustration of a repeater and its connections is shown below:

.. aafig::
    :textual:

                     +------------------------------------+
                     |          "Repeater"                |
    -------------+   |       +--------------------+       |   +---
    "Entangling" |   |   qin1| "QuantumProcessor" | qin0  |   |
    "Connection" O---O--->---O                    O---<---O---O
                 |   |       |                    |       |   |
    -------------+   |       +--------------------+       |   +---
                     |                                    |
    -------------+   |                                    |   +---
    "Classical"  |   |  "Forward to R"                    |   |
    "Connection" O---O->-# - - - - - - - - - - - - - - - -O---O
                 |   |"ccon_L"                    "ccon_R"|   |
    -------------+   +------------------------------------+   +---

We will re-use the Connection subclasses
:py:class:`~netsquid.examples.teleportation.EntanglingConnection` and :py:class:`~netsquid.examples.teleportation.ClassicalConnection`
created in the `teleportation tutorial <tutorial.simulation.html>`_ and
use the following function to create quantum processors for each node:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: create_qprocessor

We create a network component and add the nodes and connections to it.
This way we can easily keep track of all our components in the network, which will be useful when collecting data later.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: setup_network

We have used a custom noise model in this example, which helps to exaggerate the effectiveness of the repeater chain.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: FibreDepolarizeModel

The next step is to setup the protocols.
To easily manage all the protocols, we add them as subprotocols of one main protocol.
In this way, we can start them at the same time, and the main protocol will stop when all subprotocols have finished.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: setup_repeater_protocol

The definition of the swapping subprotocol that will run on each repeater node:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: SwapProtocol

The definition of the correction subprotocol responsible for applying the classical corrections at Bob:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: CorrectProtocol

Note that for the corrections we only need to apply each of the :math:`X` or :math:`Z` operators maximally once.
This is due to the anti-commutativity and self-inverse of these Pauli matrices, i.e.
:math:`ZX = -XZ` and :math:`XX=ZZ=I`, which allows us to cancel repeated occurrences up to a global phase (-1).
The program that executes the correction on Bob's quantum processor is:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: SwapCorrectProgram

With our network and protocols ready, we can add a data collector to define when and which data we want to collect.
We can wait for the signal sent by the *CorrectProtocol* when it finishes, and if it has, compute the fidelity of the
qubits at Alice and Bob with respect to the expected Bell state.
Using our network and main protocol we can easily find Alice and the CorrectionProtocol as subcomponents.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: setup_datacollector

We want to run the experiment for multiple numbers of nodes and distances.
Let us first define a function to run a single simulation:

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: run_simulation

Finally we will run multiple simulations and plot them in a figure.

.. literalinclude:: ../../netsquid/examples/repeater_chain.py
    :pyobject: create_plot

Using the ket vector formalism produces the following figure:

.. image:: ../_static/rep_chain_example_ket.png
    :width: 600px
    :align: center

Because the quantum states don't have an opportunity to grow very large in our simulation,
it is also possible for this simple example to improve on our results using the density matrix formalism.
Instead of running the simulation for 2000 iterations, it is now sufficient to run it for only a few.
As we see in the figure below, the error-bars now become negligible:

.. image:: ../_static/rep_chain_example_dm.png
    :width: 600px
    :align: center

"""
import pandas
import pydynaa
import numpy as np

import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel, QuantumErrorModel
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z
from netsquid.nodes import Node, Network
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.util.datacollector import DataCollector
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.node import Node
from netsquid.nodes.network import Network
from netsquid.examples.entanglenodes import EntangleNodes
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate, INSTR_MEASURE_BELL, INSTR_X, INSTR_Z
from netsquid.components import instructions as instr
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.nodes.connections import DirectConnection
from pydynaa import EventExpression
__all__ = [
    "SwapProtocol",
    "SwapCorrectProgram",
    "CorrectProtocol",
    "FibreDepolarizeModel",
    "create_qprocessor",
    "setup_network",
    "setup_repeater_protocol",
    "setup_datacollector",
    "run_simulation",
    "create_plot",
]

#in this program I'm adding filtering and distiling as extra
#stick to filter
class Filter(NodeProtocol):
    """Protocol that does local filtering on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    start_expression : :class:`~pydynaa.core.EventExpression` or None, optional
        Event expression node should wait for before starting filter.
        This event expression should have a
        :class:`~netsquid.protocols.protocol.Protocol` as source and should by fired
        by signalling a signal by this protocol, with the position of the qubit on the
        quantum memory as signal result.
        Must be set before the protocol can start
    msg_header : str, optional
        Value of header meta field used for classical communication.
    epsilon : float, optional
        Parameter used in filter's measurement operator.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    Attributes
    ----------
    meas_ops : list
        Measurement operators to use for filter general measurement.

    """

    def __init__(self, node, port, start_expression=None, msg_header="filter",
                 epsilon=0.3, name=None):
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "Filter({}, {})".format(node.name, port.name)
        super().__init__(node, name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self.local_qcount = 0
        self.local_meas_OK = False
        self.remote_qcount = 0
        self.remote_meas_OK = False
        self.header = msg_header
        self._qmem_pos = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self._set_measurement_operators(epsilon)

    def _set_measurement_operators(self, epsilon):
        m0 = ops.Operator("M0", np.sqrt(epsilon) * outerprod(s0) + outerprod(s1))
        m1 = ops.Operator("M1", np.sqrt(1 - epsilon) * outerprod(s0))
        self.meas_ops = [m0, m1]

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_OK = classical_message.items
                    self._handle_cchannel_rx()
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                self._qmem_pos = ready_signal.result
                yield from self._handle_qubit_rx()

    # TODO does start reset vars?
    def start(self):
        self.local_qcount = 0
        self.remote_qcount = 0
        self.local_meas_OK = False
        self.remote_meas_OK = False
        return super().start()

    def stop(self):
        super().stop()
        # TODO should stop clear qmem_pos?
        if self._qmem_pos and self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    def _handle_qubit_rx(self):
        # Handle incoming Qubit on this node.
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # Retrieve Qubit from input store
        output = self.node.qmemory.execute_instruction(INSTR_MEASURE, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        m = output["instr"][0]
        # m = INSTR_MEASURE(self.node.qmemory, [self._qmem_pos], meas_operators=self.meas_ops)[0]
        self.local_qcount += 1
        self.local_meas_OK = (m == 0)
        self.port.tx_output(Message([self.local_qcount, self.local_meas_OK], header=self.header))
        self._check_success()

    def _handle_cchannel_rx(self):
        # Handle incoming classical message from sister node.
        if (self.local_qcount == self.remote_qcount and
                self._qmem_pos is not None and
                self.node.qmemory.mem_positions[self._qmem_pos].in_use):
            self._check_success()

    def _check_success(self):
        # Check if protocol succeeded after receiving new input (qubit or classical information).
        # Returns true if protocol has succeeded on this node
        if (self.local_qcount > 0 and self.local_qcount == self.remote_qcount and
                self.local_meas_OK and self.remote_meas_OK):
            # SUCCESS!
            self.send_signal(Signals.SUCCESS, self._qmem_pos)
        elif self.local_meas_OK and self.local_qcount > self.remote_qcount:
            # Need to wait for latest remote status
            pass
        else:
            # FAILURE
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)

    def _handle_fail(self):
        if self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 1:
            return False
        return True

#In distill nothing as been changed
class Distil(NodeProtocol):
    """Protocol that does local DEJMPS distillation on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    role : "A" or "B"
        Distillation requires that one of the nodes ("B") conjugate its rotation,
        while the other doesn't ("A").
    start_expression : :class:`~pydynaa.EventExpression`
        EventExpression node should wait for before starting distillation.
        The EventExpression should have a protocol as source, this protocol should signal the quantum memory position
        of the qubit.
    msg_header : str, optional
        Value of header meta field used for classical communication.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """
    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    def __init__(self, node, port, role, start_expression=None, msg_header="distil", name=None):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self._program = self._setup_dejmp_program(conj_rotation)
        # self.INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_positions = [None, None]
        self._waiting_on_second_qubit = False
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def _setup_dejmp_program(self, conj_rotation):
        INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_ROT, [q1])
        prog.apply(INSTR_ROT, [q2])
        prog.apply(INSTR_CNOT, [q1, q2])
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False)
        return prog

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.port.rx_input(header=self.header)
                if classical_message:
                    self.remote_qcount, self.remote_meas_result = classical_message.items
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                yield from self._handle_new_qubit(ready_signal.result)
            self._check_success()

    def start(self):
        # Clear any held qubits
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        return super().start()

    def _clear_qmem_positions(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]

    def _handle_new_qubit(self, memory_position):
        # Process signalling of new entangled qubit
        assert not self.node.qmemory.mem_positions[memory_position].is_empty
        if self._waiting_on_second_qubit:
            # Second qubit arrived: perform distil
            assert not self.node.qmemory.mem_positions[self._qmem_positions[0]].is_empty
            assert memory_position != self._qmem_positions[0]
            self._qmem_positions[1] = memory_position
            self._waiting_on_second_qubit = False
            yield from self._node_do_DEJMPS()
        else:
            # New candidate for first qubit arrived
            # Pop previous qubit if present:
            pop_positions = [p for p in self._qmem_positions if p is not None and p != memory_position]
            if len(pop_positions) > 0:
                self.node.qmemory.pop(positions=pop_positions)
            # Set new position:
            self._qmem_positions[0] = memory_position
            self._qmem_positions[1] = None
            self.local_qcount += 1
            self.local_meas_result = None
            self._waiting_on_second_qubit = True

    def _node_do_DEJMPS(self):
        # Perform DEJMPS distillation protocol locally on one node
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # We perform local DEJMPS
        yield self.node.qmemory.execute_program(self._program, [pos1, pos2])  # If instruction not instant
        self.local_meas_result = self._program.output["m"][0]
        self._qmem_positions[1] = None
        # Send local results to the remote node to allow it to check for success.
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))

    def _check_success(self):
        # Check if distillation succeeded by comparing local and remote results
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                # SUCCESS
                self.send_signal(Signals.SUCCESS, self._qmem_positions[0])
            else:
                # FAILURE
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
            self.local_meas_result = None
            self.remote_meas_result = None
            self._qmem_positions = [None, None]

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 2:
            return False
        return True

class SwapProtocol(NodeProtocol):
    """Perform Swap on a repeater node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    name : str
        Name of this protocol.

    """

    def __init__(self, node, name):
        super().__init__(node, name)
        self._qmem_input_port_l = self.node.qmemory.ports["qin1"]
        self._qmem_input_port_r = self.node.qmemory.ports["qin0"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):
        while True:
            yield (self.await_port_input(self._qmem_input_port_l) &
                   self.await_port_input(self._qmem_input_port_r))
            # Perform Bell measurement
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1, 0])
            m, = self._program.output["m"]
            # Send result to right node on end
            self.node.ports["ccon_R"].tx_output(Message(m))

            

class SwapCorrectProgram(QuantumProgram):
    """Quantum processor program that applies all swap corrections."""
    default_num_qubits = 1

    def set_corrections(self, x_corr, z_corr):
        self.x_corr = x_corr % 2
        self.z_corr = z_corr % 2

    def program(self):
        q1, = self.get_qubit_indices(1)
        if self.x_corr == 1:
            self.apply(INSTR_X, q1)
        if self.z_corr == 1:
            self.apply(INSTR_Z, q1)
        yield self.run()

class CorrectProtocol(NodeProtocol):
    """Perform corrections for a swap on an end-node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    num_nodes : int
        Number of nodes in the repeater chain network.

    """

    def __init__(self, node, num_nodes):
        super().__init__(node, "CorrectProtocol")
        self.num_nodes = num_nodes
        self._x_corr = 0
        self._z_corr = 0
        self._program = SwapCorrectProgram()
        self._counter = 0

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["ccon_L"])
            message = self.node.ports["ccon_L"].rx_input()
            if message is None or len(message.items) != 1:
                continue
            m = message.items[0]
            if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                self._x_corr += 1
            if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                self._z_corr += 1
            self._counter += 1
            if self._counter == self.num_nodes - 2:
                if self._x_corr or self._z_corr:
                    self._program.set_corrections(self._x_corr, self._z_corr)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1])
                self.send_signal(Signals.SUCCESS)
                self._x_corr = 0
                self._z_corr = 0
                self._counter = 0

def create_qprocessor(name, links):
    """Factory to create a quantum processor for each node in the repeater chain network.

    Has two memory positions and the physical instructions necessary for teleportation.

    Parameters
    ----------
    name : str
        Name of the quantum processor.

    Returns
    -------
    :class:`~netsquid.components.qprocessor.QuantumProcessor`
        A quantum processor to specification.

    """
    noise_rate = 200
    gate_duration = 1
    gate_noise_model = DephaseNoiseModel(noise_rate)
    mem_noise_model = DepolarNoiseModel(noise_rate)
    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration=gate_duration,
                            quantum_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_Z, duration=gate_duration,
                            quantum_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=gate_duration),
    ]
    #fallback to nonphysical adds decoherence
    qproc = QuantumProcessor(name, num_positions=2**(links+1), fallback_to_nonphysical=False,
                             mem_noise_models=[mem_noise_model] * (2**(links+1)), #this defines the number of quantum memory
                             phys_instructions=physical_instructions)
    return qproc

class PurifyProtocol(LocalProtocol):
    r"""Here we want to just call purify This should only call for the instructions.

    Combines the sub-protocols:
    - :py:class:`~netsquid.examples.entanglenodes.EntangleNodes`
    - :py:class:`~netsquid.examples.purify.Filter`

    Will run for specified number of times then stop, recording results after each run.

    Parameters
    ----------
    node_a : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.

    node_b : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    num_runs : int
        Number of successful runs to do.
    epsilon : float
        Parameter used in filter's measurement operator.

    Attributes
    ----------
    results : :py:obj:`dict`
        Dictionary containing results. Results are :py:class:`numpy.array`\s.
        Results keys are *F2*, *pairs*, and *time*.

    Subprotocols
    ------------
    entangle_A : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node A.
    entangle_B : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node B.
    purify_A : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node A.
    purify_B : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node B.

    Notes
    -----
        The filter purification does not support the stabilizer formalism.

    """
    #here we define the attributes of the object
    #wait for swapping signal
    def __init__(self, node_left, node_right, num_runs, epsilon=0.3):
        super().__init__(nodes={"A": node_left, "B": node_right}, name="Filter")
        self._epsilon = epsilon
        self.num_runs = num_runs        
        # Initialise sub-protocols
        #Q: should the entanglement be called again in here for the class or should I take entanglingconnection?
        #here call entanglement should we
        #name if entangle has been modified
        self.add_subprotocol(EntangleNodes(node=node_left, role="source", input_mem_pos=0,
                                           num_pairs=2, name="entangle_L"))
        self.add_subprotocol(EntangleNodes(node=node_right, role="receiver", input_mem_pos=0,
                                           num_pairs=2, name="entangle_R"))
        # self.add_subprotocol(Distil(node_left, node_left.ports["ccon_R"],
        #                             role="A", name="purify_L"))
        # self.add_subprotocol(Distil(node_right, node_right.ports["ccon_L"],
        #                             role="B", name="purify_R"))

        self.add_subprotocol(Filter(node_left, node_left.ports["ccon_R"],start_expression=None,
                                    epsilon=epsilon, name="purify_L"))
        print("hi","Type of purify_L start_expression:", type(self.subprotocols["purify_L"].start_expression))

        self.add_subprotocol(Filter(node_right, node_right.ports["ccon_L"],start_expression=None,
                                    epsilon=epsilon, name="purify_R"))
        print("Debug: Added subprotocols:", list(self.subprotocols.keys()))

        #A=L and B=R
        #Problem comes from filtering expecting a signal (start expression)
        #Set start expressions
        #tell this program to send signal success
        self.subprotocols["purify_L"].start_expression = (
            self.subprotocols["purify_L"].await_signal(self.subprotocols["entangle_L"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_R"].start_expression = (
            self.subprotocols["purify_R"].await_signal(self.subprotocols["entangle_R"],
                                                       Signals.SUCCESS))
        astart_expression = (self.subprotocols["entangle_L"].await_signal(
                            self.subprotocols["purify_L"], Signals.FAIL) |
                            self.subprotocols["entangle_L"].await_signal(
                                self, Signals.WAITING))
        print("debugstatement")
        self.subprotocols["entangle_L"].start_expression = astart_expression
        self.send_signal(Signals.SUCCESS)
        print("node_left:", node_left)
        print("node_left.ports['ccon_R']:", node_left.ports.get("ccon_R"))
        print("epsilon:", epsilon)
        print("debugging:")
        try:
            filter_L = Filter(node_left, node_left.ports["ccon_R"],start_expression=astart_expression)
            filter_R = Filter(node_right, node_right.ports["ccon_L"],start_expression=astart_expression)
            print("Filter_L and Filter_R initialized")
        except Exception as e:
            print("Error during Filter initialization:", e)


    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            self.subprotocols["entangle_L"].entangled_pairs = 0
            self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["purify_L"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_R"], Signals.SUCCESS))
            signal_L = self.subprotocols["purify_L"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_R = self.subprotocols["purify_R"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result = {
                "pos_L": signal_L,
                "pos_R": signal_R,
                "time": sim_time() - start_time,
                "pairs": self.subprotocols["entangle_L"].entangled_pairs,
            }
            self.send_signal(Signals.SUCCESS, result)


def setup_network(num_nodes, node_distance, source_frequency):
    """Setup repeater chain network.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the network, at least 3.
    node_distance : float
        Distance between nodes [km].
    source_frequency : float
        Frequency at which the sources create entangled qubits [Hz].

    Returns
    -------
    :class:`~netsquid.nodes.network.Network`
        Network component with all nodes and connections as subcomponents.

    """
    if num_nodes < 3:
        raise ValueError(f"Can't create repeater chain with {num_nodes} nodes.")
    network = Network("Repeater_chain_network")
    # Create nodes with quantum processors
    nodes = []
    nodes_copy = []
    nestlvl=int(np.log2(num_nodes))
    #here nesting levels have been added and num zeros modified
    for i in range(num_nodes):
        # Prepend leading zeros to the number
        num_zeros = int(np.log10(num_nodes)) + 1
        nodes.append(Node(f"Node_{i:0{num_zeros}d}", qmemory=create_qprocessor(f"qproc_{i}",nestlvl)))
    network.add_nodes(nodes)
    # Create quantum and classical connections:
    #add subcomponents
    for i in range(num_nodes - 1):
        node_left, node_right = nodes[i], nodes[i + 1]
        # Create quantum connection
        qconn = EntanglingConnection(name=f"qconn_{i}-{i+1}", length=node_distance,
                                     source_frequency=source_frequency)
        #this is to give roles to network
        # node_left.add_subcomponents(Qsource("QSource_L", StateSampler=StateSampler))
        # Add a noise model which depolarizes the qubits exponentially
        # depending on the connection length
        for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
            qconn.subcomponents[channel_name].models['quantum_noise_model'] =\
                FibreDepolarizeModel()
        port_name, port_r_name = network.add_connection(
            node_left, node_right, connection=qconn, label="quantum")
        # Forward qconn directly to quantum memories for right and left inputs:
        node_left.ports[port_name].forward_input(node_left.qmemory.ports["qin0"])  # R input
        node_right.ports[port_r_name].forward_input(node_right.qmemory.ports["qin1"])  # L input
        # Create classical connection
        cconn = ClassicalConnection(name=f"cconn_{i}-{i+1}", length=node_distance)
        port_name, port_r_name = network.add_connection(
            node_left, node_right, connection=cconn, label="classical",
            port_name_node1="ccon_R", port_name_node2="ccon_L")
        print ("L:",port_name,"  R:",port_r_name)
        #Forward cconn to right most node
        if "ccon_L" in node_left.ports:
            node_left.ports["ccon_L"].bind_input_handler(
                lambda message, _node=node_left: _node.ports["ccon_R"].tx_output(message))
    return network


def setup_repeater_protocol(network):
    """Setup repeater protocol on repeater chain network.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        Repeater chain network to put protocols on.

    Returns
    -------
    :class:`~netsquid.protocols.protocol.Protocol`
        Protocol holding all subprotocols used in the network.

    """
    protocol = LocalProtocol(nodes=network.nodes)
    # Add SwapProtocol to all repeater nodes. Note: we use unique names,
    # since the subprotocols would otherwise overwrite each other in the main protocol.
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
    nodes_copy=nodes.copy() #a copy of the original nodes
    while len(nodes_copy)>2: #instructions until nodes is only two
        swapped_nodes=[]
        for i in range(len(nodes_copy)):
            if (i % 2)==1:
                subprotocol = SwapProtocol(node=nodes_copy[i], name=f"Swap_{nodes_copy[i].name}") #swap protocol
                protocol.add_subprotocol(subprotocol)
                filter = PurifyProtocol(node_left=nodes_copy[i], node_right=nodes_copy[i+1], num_runs=100, epsilon=0.3)
                protocol.add_subprotocol(filter)
                swapped_nodes.append(nodes_copy[i]) #stores in the empty list
        for swapped in swapped_nodes:
                nodes_copy.remove(swapped) 
        print('checking')
    #if we modify this to nodes_copy it stops storing fidelity
    # Add CorrectProtocol to Bob
    subprotocol = CorrectProtocol(nodes[-1], len(nodes))
    protocol.add_subprotocol(subprotocol)
    return protocol


def setup_datacollector(network, protocol):
    """Setup the datacollector to calculate the fidelity
    when the CorrectionProtocol has finished.

    Parameters
    ----------
    network : :class:`~netsquid.nodes.network.Network`
        Repeater chain network to put protocols on.

    protocol : :class:`~netsquid.protocols.protocol.Protocol`
        Protocol holding all subprotocols used in the network.

    Returns
    -------
    :class:`~netsquid.util.datacollector.DataCollector`
        Datacollector recording fidelity data.

    """
    # Ensure nodes are ordered in the chain:
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
    nodes_copy=nodes.copy
    def calc_fidelity(evexpr):
        qubit_a, = nodes[0].qmemory.peek([0])
        qubit_b, = nodes[-1].qmemory.peek([1])
        fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True)
        #print('t',ns.sim_time())\
        return {"fidelity": fidelity}
    

    dc = DataCollector(calc_fidelity, include_entity_name=False)
    dc.collect_on(pydynaa.EventExpression(source=protocol.subprotocols['CorrectProtocol'],
                                          event_type=Signals.SUCCESS.value))
    
    return dc


def run_simulation(num_nodes=7, node_distance=20, num_iters=100):
    """Run the simulation experiment and return the collected data.

    Parameters
    ----------
    num_nodes : int, optional
        Number nodes in the repeater chain network. At least 3. Default 4.
    node_distance : float, optional
        Distance between nodes, larger than 0. Default 20 [km].
    num_iters : int, optional
        Number of simulation runs. Default 100.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe with recorded fidelity data.

    """
    ns.sim_reset()
    est_runtime = (0.5 + num_nodes - 1) * node_distance * 5e3
    network = setup_network(num_nodes, node_distance=node_distance,
                            source_frequency=1e9 / est_runtime)
    protocol = setup_repeater_protocol(network)
    dc = setup_datacollector(network, protocol)
    protocol.start()
    ns.sim_run(est_runtime * num_iters)
    return dc.dataframe


def create_plot(num_iters=2000):
    """Run the simulation for multiple nodes and distances and show them in a figure.

    Parameters
    ----------

    num_iters : int, optional
        Number of iterations per simulation configuration.
        At least 1. Default 2000.
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    for distance in [10, 30, 50]:
        data = pandas.DataFrame()
        nesting_levels=list(range(1,7))
        num_nodes = [int(2**nesting_level+1) for nesting_level in nesting_levels]
        for num_node in num_nodes:
            print(f"======  num_node: {num_node}    distance: {distance}  =========")
            data[num_node] = run_simulation(num_nodes=num_node,
                                            node_distance=distance / num_node,
                                            num_iters=num_iters)['fidelity']
        # For errorbars we use the standard error of the mean (sem)
        data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
        data.plot(y='fidelity', yerr='sem', label=f"{distance} km", ax=ax)
    plt.xlabel("number of nodes")
    plt.ylabel("fidelity")
    plt.title("Repeater chain with different total lengths")
    plt.show()


class FibreDepolarizeModel(QuantumErrorModel):
    """Custom non-physical error model used to show the effectiveness
    of repeater chains.

    The default values are chosen to make a nice figure,
    and don't represent any physical system.

    Parameters
    ----------
    p_depol_init : float, optional
        Probability of depolarization on entering a fibre.
        Must be between 0 and 1. Default 0.009
    p_depol_length : float, optional
        Probability of depolarization per km of fibre.
        Must be between 0 and 1. Default 0.025

    """

    def __init__(self, p_depol_init=0.009, p_depol_length=0.025):
        super().__init__()
        self.properties['p_depol_init'] = p_depol_init
        self.properties['p_depol_length'] = p_depol_length
        self.required_properties = ['length']

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Uses the length property to calculate a depolarization probability,
        and applies it to the qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Not used.

        """
        for qubit in qubits:
            prob = 1 - (1 - self.properties['p_depol_init']) * np.power(
                10, - kwargs['length']**2 * self.properties['p_depol_length'] / 10)
            ns.qubits.depolarize(qubit, prob=prob)


if __name__ == "__main__":
    ns.set_qstate_formalism(ns.QFormalism.DM)
    create_plot(20)
