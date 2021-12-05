# Improved Partition Optimization

This project contains the sources for an optimization approach based on improved partitions.
This work uses IBMs [Qiskit](https://qiskit.org/) and presents a new passmanager implementing the new optimization approach.

## Structure
The project contains sources of the implementation for a passmanager based on improved partitioning.
It is defined as [passmanager_SA](https://github.com/mnm-admin/improvedPartitionOptimization/blob/main/Passmanager/passmanager_SA.py).
Together with other passmanagers used for comparison, it can be found in the folder [Passmanager](https://github.com/mnm-admin/improvedPartitionOptimization/tree/main/Passmanager).

Simulated Annealing is used to create an improved partition for a given quantum circuit.
In folder [Passes](https://github.com/mnm-admin/improvedPartitionOptimization/tree/main/Passes) all additional passes are listed.
These include:
- [collectAndDecompose](https://github.com/mnm-admin/improvedPartitionOptimization/blob/main/Passes/collectAndDecompose.py)
- [evaluate](https://github.com/mnm-admin/improvedPartitionOptimization/blob/main/Passes/evaluate.py)
- [partitioning](https://github.com/mnm-admin/improvedPartitionOptimization/blob/main/Passes/partitioning.py)
- [partitioning_init](https://github.com/mnm-admin/improvedPartitionOptimization/blob/main/Passes/partitioning_init.py)
- [transformToBasis](https://github.com/mnm-admin/improvedPartitionOptimization/blob/main/Passes/transformToBasis.py)

For testing purposes a [jupyter notebook](https://github.com/mnm-admin/improvedPartitionOptimization/blob/main/Example%20Evaluation.ipynb) is presented. 
It contains the sources to optimize quantum circuits contained in folders [LargeCircuits](https://github.com/mnm-admin/improvedPartitionOptimization/tree/main/LargeCircuits) and [SmallCircuits](https://github.com/mnm-admin/improvedPartitionOptimization/tree/main/SmallCircuits).
This example will saves all evaluation results in a folder called "Logs".

## Testing Circuits
To verify the new approach a set of different quantum circuits were used.

### Circuit generator
To test the optimization approach random circuits were used.
The generator method creates such circuits and can be configured to individual demands.


### Qasm Benchmark

For further benchmarking a subset of an existing set of quantum circuits was used.
The circuits are available as [QASMBench](https://github.com/pnnl/QASMBench) and are described in detail inside their [paper](https://github.com/pnnl/QASMBench/blob/master/qasmbench.pdf)(also available over [arXiv](https://arxiv.org/abs/2005.13018))


### Small Circuits

Five circuits were defined to evaluate the behaviour of the optimization approach.
They contain a maximum of three qubits.
