from qiskit.transpiler.basepasses import AnalysisPass


class Cost(AnalysisPass):
    """Calculate the cost of a DAG circuit.
    The costs consist of the amount of gates and their given errors."""
    
    def run(self, dag):
        """Run the Cost pass on `dag`."""
        #print("running")
        ops = dag.count_ops()
        err_conf = {'id':1,'rz':1, 'sx':1, 'x':1, 'cx':26}
        alpha = 1
        
        total_cost = 0
        for k in ops:
            if k in err_conf.keys():                
                total_cost = total_cost + err_conf[k]*alpha*ops[k]
            else:
                # key not specified, use default 1
                total_cost = total_cost + 1*alpha*ops[k]

        self.property_set['cost'] = total_cost