import gurobipy as gp
from gurobipy import GRB
from typing import List

class MIPProblem:
    """
    Represents a Mixed-Integer Programming problem.
    It creates a single, configured Gurobi environment and loads the model into it.
    This ensures all subsequent operations (relax, copy) inherit the correct parameters.
    """
    def __init__(self, file_path: str):
        # --- NEW: CREATE AND CONFIGURE THE MASTER ENVIRONMENT ---
        # Create a new environment. All models created within this 'with' block
        # will use this environment and inherit its settings.
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0) # Suppress all default Gurobi output
            env.start()

            # --- OUR CORE SETTINGS ---
            # Now, load the model using this pre-configured environment.
            self.model: gp.Model = gp.read(file_path, env=env)

            # Set the parameters for how all subsequent LPs should be solved.
            # We do this ONCE on the main model. Copies and relaxations will inherit this.
            self.model.setParam('Presolve', 0)
            self.model.setParam('Cuts', 0)
            self.model.setParam('Heuristics', 0)
            self.model.setParam(GRB.Param.Method, 0) # Primal Simplex
        # --- END OF NEW LOGIC ---

        self.integer_variable_names: List[str] = []
        for v in self.model.getVars():
            if v.VType in [GRB.INTEGER, GRB.BINARY]:
                self.integer_variable_names.append(v.VarName)

    def remove_constraints_by_index(self, constr_indices: List[int]):
        """
        Removes constraints from the model by their indices.
        It's crucial to sort indices in descending order to avoid re-indexing
        issues during removal.
        """
        all_constrs = self.model.getConstrs()
        constrs_to_remove = [all_constrs[i] for i in sorted(constr_indices, reverse=True)]
        self.model.remove(constrs_to_remove)
        self.model.update()