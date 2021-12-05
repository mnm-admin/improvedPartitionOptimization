import logging

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit

from qiskit.circuit.library.standard_gates import (iSwapGate, CXGate, CZGate,
                                                   RXXGate, ECRGate)
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer

from qiskit.quantum_info.synthesis import one_qubit_decompose


from qiskit.extensions import UnitaryGate

LOG = logging.getLogger(__name__)



def _choose_kak_gate(basis_gates):
    """Choose the first available 2q gate to use in the KAK decomposition."""

    kak_gate_names = {
        'cx': CXGate(),
        'cz': CZGate(),
        'iswap': iSwapGate(),
        'rxx': RXXGate(np.pi / 2),
        'ecr': ECRGate()
    }

    kak_gate = None
    kak_gates = set(basis_gates or []).intersection(kak_gate_names.keys())
    if kak_gates:
        kak_gate = kak_gate_names[kak_gates.pop()]

    return kak_gate

def _choose_euler_basis(basis_gates):
    """"Choose the first available 1q basis to use in the Euler decomposition."""
    basis_set = set(basis_gates or [])

    for basis, gates in one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES.items():
        if set(gates).issubset(basis_set):
            return basis

    return None

def multGlobalPhaseCX(unitary):
    sol1 = np.complex128(unitary[0][0]) + np.complex128(unitary[0][1]) + np.complex128(unitary[0][2]) + np.complex128(unitary[0][3])
    sol2 = np.complex128(unitary[1][0]) + np.complex128(unitary[1][1]) + np.complex128(unitary[1][2]) + np.complex128(unitary[1][3])
    sol3 = np.complex128(unitary[2][0]) + np.complex128(unitary[2][1]) + np.complex128(unitary[2][2]) + np.complex128(unitary[2][3])
    sol4 = np.complex128(unitary[3][0]) + np.complex128(unitary[3][1]) + np.complex128(unitary[3][2]) + np.complex128(unitary[3][3])

    if (sol1 != 1 and sol1 == sol2 and sol2 == sol3 and sol3==sol4):
        return  unitary / sol1
    else:
        return unitary


class OptimizeCollectAndDecompose(TransformationPass):
    def __init__(self, basis=None):
        super().__init__()
        

    def run(self, dag):

        # TODO: Check was der counter nochmal soll und ob der hier gebraucht wird...
        #if self.property_set['counter'] != None:
        #    self.property_set['counter'] = self.property_set['counter'] + 1
        #else:
        #    self.property_set['counter'] = 0
        eps=1e-15
        basis_gates = ['id','rz', 'sx', 'x', 'cx']
        euler_basis = _choose_euler_basis(basis_gates)
        kak_gate = _choose_kak_gate(basis_gates)       
        decomposer2q = TwoQubitBasisDecomposer(kak_gate, euler_basis=euler_basis, basis_fidelity=1)
        
        state = self.property_set['state']     
        singleGates = self.property_set['singleGates']
        res_circ = QuantumCircuit(len(dag.wires))
        
        # Add one Qubit runs to resulting circuit
        for wire in singleGates:
            # add wire
            for node in dag.nodes_on_wire(wire , True):
                res_circ.append(node.op, [res_circ.qubits[wire.index]])

            
        res_dag = circuit_to_dag(res_circ)
        # Collect and decompose blocks
        for block in state:
            Xa, Xb, startA, startB, endA, endB = block 

            nodesA = list(dag.nodes_on_wire(Xa, True))[startA:endA]
            nodesB = list(dag.nodes_on_wire(Xb, True))[startB:endB]

            # Create subcircuit containing affected gates
            subcirc = QuantumCircuit(2) #
            count_a = 0
            count_b = 0
            la = endA - startA
            lb = endB - startB

            while(count_a < la or count_b < lb):
                cx = False
                if count_a < la:
                    node = nodesA[count_a]
                    if node.name != 'cx':
                        subcirc.append(node.op,[0])
                        count_a = count_a + 1
                    else:
                        cx = True               
                if count_b < lb:
                    node = nodesB[count_b]
                    
                    if node.name != 'cx':
                        subcirc.append(node.op,[1])                        
                        cx = False
                        count_b = count_b + 1
               
                if cx:
                    if node.qargs[0].index == Xa.index:
                        # control is 0, target 1
                        subcirc.append(node.op,[0,1])
                    else:
                        # control is 1, target 0
                        subcirc.append(node.op,[1,0])
                    count_a = count_a + 1
                    count_b = count_b + 1

            unitary = UnitaryGate(Operator(subcirc))  # simulates the circuit
            uniMat = unitary.to_matrix()
            
            # Remove values smaller epsilon, remove factor and unnecessary signs
            uniMat.real[abs(uniMat.real) < eps] = 0.0
            uniMat.imag[abs(uniMat.imag) < eps] = 0.0
            uniMat= multGlobalPhaseCX(uniMat)
            uniMat.real[abs(uniMat.real) == 0] = 0.0
            uniMat.imag[abs(uniMat.imag) == 0] = 0.0
            
            # Decompose matrix
            new_circ = decomposer2q(uniMat)
            # Keep old circ of resulting new circuit is larger than old one
            if sum(circuit_to_dag(subcirc).count_ops().values()) > sum(circuit_to_dag(new_circ).count_ops().values()):
                new_dag = circuit_to_dag(new_circ)
            else:
                new_dag = circuit_to_dag(subcirc)
            # Re-assemble whole circuit
            res_dag.compose(new_dag, qubits=[Xa.index,Xb.index])
        return res_dag 