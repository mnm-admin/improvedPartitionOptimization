
from copy import deepcopy

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import AnalysisPass
from datetime import datetime

class Evaluate(AnalysisPass):
    def convert(seconds):
        min, sec = divmod(seconds, 60)
        hour, min = divmod(min, 60)
        return "%d:%02d:%02d" % (hour, min, sec)

    def __init__(self, logPath):
        super().__init__()
        self.logPath = logPath

    def run(self, dag):
        # Get amount of gates used
        ops = dag.count_ops()        
        # Safe current graph status to file
        with open(self.logPath, "a") as f:
            f.write(str(sum(dag.count_ops().values())) + ";" + str(ops) + ";" + str(datetime.now()) + "\n")
            f.close()