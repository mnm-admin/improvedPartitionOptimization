import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from qiskit.transpiler import TranspilerError

# import new passes
from Passes.transformToBasis import transformToBasis
from Passes.partitioning_init import SelectUnitaryBlocks_init
from Passes.collectAndDecompose import OptimizeCollectAndDecompose
from Passes.cost import Cost
from Passes.evaluate import Evaluate


import os

# Only one basis gate set supportet yet 
basis_gates = ['id','rz', 'sx', 'x', 'cx']

def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%d:%02d:%02d" % (hour, min, sec)

def testing_pass_manager_init( logFile, singleIt=False) -> PassManager:  

    def _opt_control(property_set):
        return not property_set['cost_fixed_point'] 
    
    # Condition for loop
    _cost_check = [Cost(), FixedPoint('cost')]
    
    # Unroll to basis gates?
    _to_basis = [Unroll3qOrMore(), transformToBasis(basis_gates)]
    
    # Build blocks by collection strategie and decompose them
    _select_blocks = [SelectUnitaryBlocks_init()]
    _collect_decompose = [OptimizeCollectAndDecompose()]
    

    # Logging path
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    _evaluation = [Evaluate(logFile)]

    # Single qubit optimization
    _one_qubit_opt = [Optimize1qGatesDecomposition(basis_gates)] #Aktuell den, später vielleicht einen anderen
    
    # Build pass manager
    pm = PassManager()
    pm.append(_evaluation + _to_basis + _evaluation) #Reicht das zum decomposen in basis gates? Evt eigenen Pass schreiben...

    # Single iteration or based on condition
    if singleIt:
        pm.append(_cost_check + _select_blocks + _collect_decompose + _evaluation + _one_qubit_opt)
    else:
        pm.append(_cost_check + _select_blocks + _collect_decompose + _evaluation + _one_qubit_opt, do_while=_opt_control)
      
    return pm