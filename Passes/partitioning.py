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


# Helper
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


class SelectUnitaryBlocks(AnalysisPass):

    
    def __init__(self, flipProbabilities, alpha, initTemp, minTemp, n, cxCost=26):
        super().__init__()
        self.flipProbabilities = flipProbabilities
        self.alpha = alpha     
        self.cxCost = cxCost
        self.initTemp = initTemp
        self.minTemp = minTemp
        self.n = n
        if sum(self.flipProbabilities) != 1:
            print("error!!!!\nTotal sum is:", sum(self.flipProbabilities))

    
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
    
    
    def _value(self,state, singleGates, dag):
        eps=1e-15
        value = 0
        basis_gates = ['id','rz', 'sx', 'x', 'cx']
        euler_basis = _choose_euler_basis(basis_gates)
        kak_gate = _choose_kak_gate(basis_gates)       
        decomposer2q = TwoQubitBasisDecomposer(kak_gate, euler_basis=euler_basis, basis_fidelity=1)
        
        res_circ = QuantumCircuit(len(dag.wires))
        # Add one Qubit runs to resulting circuit
        for wire in singleGates:
            value += len(list(dag.nodes_on_wire(wire , True)))
           
        res_dag = circuit_to_dag(res_circ)
        # Collect and decompose blocks
        for block in state:
            Xa, Xb, startA, startB, endA, endB = block # new

            # Search gates from block
            nodesA = list(dag.nodes_on_wire(Xa, True))[startA:endA]
            nodesB = list(dag.nodes_on_wire(Xb, True))[startB:endB]

            # Create subcircuit representing block
            q = QuantumRegister(2)
            subcirc = QuantumCircuit(2) 
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

            # Transform block to unitary matrix
            unitary = UnitaryGate(Operator(subcirc)) 
            uniMat = unitary.to_matrix()
            
            # Remove values smaller epsilon, remove factor and unnecessary signs
            uniMat.real[abs(uniMat.real) < eps] = 0.0
            uniMat.imag[abs(uniMat.imag) < eps] = 0.0
            uniMat= multGlobalPhaseCX(uniMat)
            uniMat.real[abs(uniMat.real) == 0] = 0.0
            uniMat.imag[abs(uniMat.imag) == 0] = 0.0           
            
            # Decompose matrix 
            tmpCirc = decomposer2q(uniMat)
            # Cals value with weighted number of gates
            ops = tmpCirc.count_ops()
            if "cx" in ops.keys():
                cx = ops["cx"]
                value += tmpCirc.size() - cx + cx * self.cxCost
            else:
                value += tmpCirc.size()
        return value
    
    def newVal(self, new_state, state, changed, v_old, dag):
        # Find value change of blocks affected by new neighbour
        eps=1e-15
        index = changed[0]
        i = changed[1]
        basis_gates = ['id','rz', 'sx', 'x', 'cx']
        euler_basis = _choose_euler_basis(basis_gates)
        kak_gate = _choose_kak_gate(basis_gates)
        decomposer2q = TwoQubitBasisDecomposer(kak_gate, euler_basis=euler_basis, basis_fidelity=1)

        old_value = 0
        new_value = 0
        # Calculate change for both affected blocks
        for j in range(2):
            # Calculate old block value
            Xa, Xb, startA, startB, endA, endB = state[index[i+j]] 

            nodesA = list(dag.nodes_on_wire(Xa, True))[startA:endA]
            nodesB = list(dag.nodes_on_wire(Xb, True))[startB:endB]
            
            # Affected gates in new subcircuit
            subcirc = QuantumCircuit(2)
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

            # Decompose matrix and safe old value
            tmpCirc = decomposer2q(uniMat)
            ops = tmpCirc.count_ops()
            if "cx" in ops.keys():
                cx = ops["cx"]
                old_value += tmpCirc.size() - cx + cx * self.cxCost
            else:
                old_value += tmpCirc.size()
            

            # Calculate new block value
            Xa, Xb, startA, startB, endA, endB = new_state[index[i+j]] # new
            nodesA = list(dag.nodes_on_wire(Xa, True))[startA:endA]
            nodesB = list(dag.nodes_on_wire(Xb, True))[startB:endB]

            # Affected gates in new subcircuit
            subcirc = QuantumCircuit(2) 
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

            # Decompose matrix and safe new value
            tmpCirc = decomposer2q(uniMat)
            ops = tmpCirc.count_ops()
            if "cx" in ops.keys():
                cx = ops["cx"]
                new_value += tmpCirc.size() - cx + cx * self.cxCost
            else:
                new_value += tmpCirc.size()
        
        # Return changed value
        return v_old + new_value - old_value 
    
    
    def _blocksOnWire(self,wire, state):
        # Find all blocks from state affecting wire 
        res = []
        index = []
        for i, block in enumerate(state):
            if block[0] == wire or block[1] == wire:
                res.append(block)
                index.append(i)
        return (res, index)
    

    def _getNeighbour(self,dag,state, blockID, chooseWire):
        diff = 0
        contacts = 0
        newLeft = None
        newRight = None
        neighbour = state.copy()
        index=None
        i = None
        if len(state) > 1:
            
            probabilities = self.flipProbabilities 
            flip = random.random()

            # Get affected blocks for contact point
            wireA, wireB, startA, startB, endA, endB = state[blockID]        
            if chooseWire == 0:
                wire = wireA
                ab = 0
            else:
                wire = wireB
                ab = 1
            blocks, index  = self._blocksOnWire(wire, neighbour)

            i = None
            b = len(blocks)
            for n in range(b-1):
                if blocks[n] == state[blockID]:
                    i = n
            # Cannot use last block because no other block follows, use -1 instead
            if i == None and b > 1:
                i == b - 1 
            
            # Check if contact point exists
            if i != None:
                rand = random.random()
                # Use the step probabilities in descending order
                # If larger step not working, use one smaller
                for step in range(len(probabilities)-1 , -1, -1):
                    if rand <= probabilities[step]:
                        if(flip >= 0.4):
                            z = -1 * step

                            # Check if moved gate is CX
                            # Check if next gate is CX 
                            valid = True
                            for k in range(1,step+1):
                                if blocks[i][4+ab] - k <= blocks[i][2+ab] or list(dag.nodes_on_wire(wire, True))[blocks[i][4+ab] - k].name == "cx":
                                    valid = False

                        else:
                            z = 1 * step

                            # Check if next gate is CX 
                            nodes = list(dag.nodes_on_wire(wire, True))

                            valid = True
                            for k in range(1,step+1):
                                if blocks[i][4+ab] + k > len(nodes) or nodes[blocks[i][4+ab] + k-1].name == "cx": #probably only check for cx enough...
                                    valid = False
                        
                        # When valid state is found, create new block definitions for state
                        if valid:
                            if (blocks[i][0] == wire):
                                newLeft = (blocks[i][0], blocks[i][1], blocks[i][2], blocks[i][3], blocks[i][4] + z, blocks[i][5])
                                t1 = blocks[i][4] + z
                            elif (blocks[i][1] == wire):
                                newLeft = (blocks[i][0], blocks[i][1], blocks[i][2], blocks[i][3], blocks[i][4], blocks[i][5] + z)
                                t1 = blocks[i][5] + z
                            else:
                                print("ERROR")
                            if (blocks[i+1][0] == wire):
                                newRight = (blocks[i+1][0], blocks[i+1][1], blocks[i+1][2] + z, blocks[i+1][3], blocks[i+1][4], blocks[i+1][5])
                                t2 = blocks[i+1][2] + z
                            elif (blocks[i+1][1] == wire):
                                newRight = (blocks[i+1][0], blocks[i+1][1], blocks[i+1][2], blocks[i+1][3] + z, blocks[i+1][4], blocks[i+1][5])
                                t2 = blocks[i+1][3] + z
                            else:
                                print("ERROR")

                            neighbour[index[i]] = newLeft
                            neighbour[index[i+1]] = newRight
                            diff += step
                            # Stop loop when neighbour found
                            break
      
        return neighbour, diff, (index, i) 

    
    
    def run(self, dag):   
        """
        
        Execute pass and create optimized partition.
        Result is written to property set
        """
        
        n = self.n
        minTemp = self.minTemp
        cooling_factor = self.alpha
        
        # Create init state 
        state, singleGates = self._initState(dag)
        # Init temperature
        temp = self.initTemp
        v_old = self._value(state, singleGates,  dag)
        while(temp > minTemp):
            # Adapt all contact points
            for blockID in range(len(state)):
                #first wire
                new_state, diff, changed = self._getNeighbour(dag, state, blockID, 0) # , initTemp, temp, alpha
                # Evaluate new_state
                if diff > 0: 
                    # Get new value
                    v_new= self.newVal(new_state, state, changed, v_old, dag) 
                    if (v_new < v_old):
                        state = new_state
                        v_old = v_new
                    else: 
                        r = random.random()
                        if  r < np.exp(-1*n*(v_new - v_old)/temp):# Choose worse new state based on probability
                            state = new_state
                            v_old = v_new
                            
                # second wire
                new_state, diff, changed = self._getNeighbour(dag, state, blockID, 1) 
                # Evaluate new_state
                if diff > 0: 
                    # Calculate new value
                    v_new= self.newVal(new_state, state, changed, v_old, dag)
                    if (v_new < v_old):
                        state = new_state
                        v_old = v_new                       
                    else: 
                        r = random.random()
                        if  r < np.exp(-1*n*(v_new - v_old)/temp): # Choose worse new state based on probability
                            state = new_state
                            v_old = v_new

            # 6. Reduce temp
            temp = temp * cooling_factor

        #  Set resulting state to global property set
        self.property_set['state'] = state
        self.property_set['singleGates'] = singleGates
        return dag