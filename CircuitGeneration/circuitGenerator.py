import random
import numpy as np
def createTestFile(gates, qubits, probs, measure=False, wholeRotate=True):
    with open('testFile.qasm', 'w') as f:
        f.write("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n")
        f.write("qreg reg[" + str(qubits) + "];\n")
        if measure:
            f.write("creg measB[" + str(qubits) + "];\n")
        
        for i in range(gates):
            rand = random.random()
            if rand < probs[0]:
                # sx
                gate = int(random.random()*qubits)
                f.write("sx reg["+str(gate)+"];\n")
            elif rand < probs[1]:
                # sx
                gate = int(random.random()*qubits)
                f.write("x reg["+str(gate)+"];\n")
            elif rand < probs[2]:
                # rz pi
                gate = int(random.random()*qubits)
                if wholeRotate:
                    f.write("rz(pi/2) reg["+str(gate)+"];\n")
                else:
                    angle = random.random() * 2 * np.pi
                    f.write("rz(" + str(angle) + ") reg["+str(gate)+"];\n")
            elif rand <= probs[3]:
                # cx
                gate1 = int(random.random()*qubits)
                gate2 = int(random.random()*qubits)
                while gate1==gate2:
                    gate1 = int(random.random()*qubits)
                    gate2 = int(random.random()*qubits)
                f.write("cx reg["+str(gate1)+"],reg["+str(gate2)+"];\n")
        
        if measure:
            for i in range(qubits):
                f.write("measure reg["+str(i)+"] -> measB[" + str(i) + "];\n")
        print("succesfully created testfile...")