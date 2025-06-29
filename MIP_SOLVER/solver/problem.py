import gurobipy as gp
from gurobipy import GRB
from typing import List

class MIPProblem:
    """
    Represents a Mixed-Integer Programming problem, loading it from a file
    and identifying integer and binary variables.
    """
    def __init__(self, file_path: str):
        """
        Initializes the MIPProblem by reading a Gurobi model from the given file path.

        Args:
            file_path (str): The path to the model file (e.g., .mps, .lp).
        """
        self.model: gp.Model = gp.read(file_path)
        self.integer_variable_names: List[str] = []

        for v in self.model.getVars():
            if v.VType == GRB.INTEGER or v.VType == GRB.BINARY:
                self.integer_variable_names.append(v.VarName)
