from collections import defaultdict
import random
import numpy as np

from qiskit.circuit import Gate
from qiskit.transpiler.basepasses import AnalysisPass

from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
from qiskit.quantum_info.synthesis import one_qubit_decompose

from qiskit.circuit.library.standard_gates import (iSwapGate, CXGate, CZGate,
                                                   RXXGate, ECRGate)

from qiskit.extensions import UnitaryGate

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit


class SelectUnitaryBlocks_init(AnalysisPass):
   
    #def __init__(self, flipProbabilities, alpha, cxCost=26):
    #    super().__init__()
    #    self.flipProbabilities = flipProbabilities
    #    self.alpha = alpha       
    #    self.cxCost = cxCost
    #    if sum(self.flipProbabilities) != 1:
    #        print("error!!!!\nTotal sum is:", sum(self.flipProbabilities))

    def __init__(self):
        super().__init__()
    
    def _initState(self,dag):
        state = []
        count = {}
        singleGates = []
        for q in dag.wires:
            count[q] = 0

        # Create wires to visit and take first
        wires = dag.wires
        wire = wires[0]

        while wires:
            # get all CX gates on wire
            nodesA = [n for n in dag.nodes_on_wire(wire, True)]
            nodesA = nodesA[count[wire]:]

            # Check if unpartitioned nodes on wire        
            if not nodesA:
                # No nodes available, remove wire
                wires.remove(wire)
                if wires:
                    wire = wires[0]
            else:
                # Unpartitioned nodes exist
                cA = []
                # Search for CX gates on wire
                for n in nodesA:
                    if n.name == "cx":
                        cA.append(n)
                # Check if qubit is connected with other qubits via CX
                if cA:
                    # CX on wire exist, set second wire 
                    # This is the selection for the next block
                    if cA[0].qargs[0] == wire:
                        wireA = cA[0].qargs[0]
                        wireB = cA[0].qargs[1]
                    else: 
                        wireA = cA[0].qargs[1]
                        wireB = cA[0].qargs[0]

                    # Search CX on second wire
                    nodesB = [n for n in dag.nodes_on_wire(wireB, True)]
                    nodesB = nodesB[count[wireB]:]
                    cB = []
                    for n in nodesB:
                        if n.name == "cx":
                            cB.append(n)
                    # Check if on both wires next CX is the same
                    if cB[0] == cA[0]:
                        # CX are the same, create block containinng this CX
                        # Add nodes from wireA to block
                        c = 0
                        counter = 0
                        for n in nodesA:
                            if n.name != "cx":
                                # Add CX acting on same qubits
                                counter = counter + 1
                            elif (set(n.qargs)==set([wireA, wireB]) and cA[c] == cB[c]):
                                # Add single qubit gates
                                counter = counter + 1
                                c = c+1
                            else:
                                # Other CX gate acting on different second qubit, block reached its end
                                break;
                        startA = count[wireA]
                        endA = startA + counter
                        count[wireA] = endA 

                        # Add nodes from wireA to block
                        counter = 0
                        c = 0
                        for n in nodesB:
                            if n.name != "cx":
                                # Add CX acting on same qubits
                                counter = counter + 1
                            elif (set(n.qargs)==set([wireA, wireB]) and cA[c] == cB[c]):
                                # Add single qubit gates
                                counter = counter + 1
                                c = c+1
                            else:
                                # Other CX gate acting on different second qubit, block reached its end
                                break;
                        startB = count[wireB]
                        endB = startB + counter
                        count[wireB] = endB 
                        # Append new block to state
                        state.append((wireA, wireB, startA, startB, endA, endB))

                    else:
                        # Second wire contains other CX before, create next block with wireB 
                        wire = wireB
                else:
                    # No cx on wire, no block created
                    count[wire] = len(nodesA)
                    singleGates.append(wire)
        return state, singleGates
    

    def run(self, dag):          
        # Create init state
        state, singleGates = self._initState(dag)

        self.property_set['state'] = state
        self.property_set['singleGates'] = singleGates

        return dag