import gurobipy as gp
from gurobipy import GRB
from typing import List

class MIPProblem:
    """
    Represents a Mixed-Integer Programming problem.
    It creates and holds a single Gurobi environment, ensuring the master model
    and all subsequent copies/relaxations remain valid.
    """
    def __init__(self, file_path: str):
        self.env: gp.Env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()

        # Loads the model using the persistent environment.
        self.model: gp.Model = gp.read(file_path, env=self.env)

        # Sets parameters that will be inherited by copies and relaxations.
        self.model.setParam('Presolve', 0)
        self.model.setParam('Cuts', 0)
        self.model.setParam('Heuristics', 0)
        self.model.setParam(GRB.Param.Method, 0) # Primal Simplex

        self.integer_variable_names: List[str] = []
        for v in self.model.getVars():
            if v.VType in [GRB.INTEGER, GRB.BINARY]:
                self.integer_variable_names.append(v.VarName)

    def remove_constraints_by_index(self, constr_indices: List[int]):
        """
        Removes constraints from the model by their indices.
        """
        all_constrs = self.model.getConstrs()
        constrs_to_remove = [all_constrs[i] for i in sorted(constr_indices, reverse=True)]
        self.model.remove(constrs_to_remove)
        self.model.update()

    def dispose(self):
        """
        A new method to properly clean up the Gurobi environment when the
        solver is completely finished.
        """
        self.env.dispose()