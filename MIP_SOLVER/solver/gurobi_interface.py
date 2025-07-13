import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Any

from solver.problem import MIPProblem
from solver.utilities import setup_logger # We'll need this for error logging

logger = setup_logger()

def solve_lp_relaxation(problem: MIPProblem, local_constraints: List[Tuple[str, str, float]]) -> Dict[str, Any]:
    """
    Solves the LP relaxation of the MIP problem.
    It now assumes the base model has already been configured with the correct parameters.
    """
    result = {'status': 'UNKNOWN'}
    relaxed_model = None # Define in this scope for the 'finally' block

    try:
        # The .relax() method will create a new model that inherits the parent's environment and parameters.
        relaxed_model = problem.model.relax()

        # Apply local branching constraints
        for var_name, sense, value in local_constraints:
            var = relaxed_model.getVarByName(var_name)
            relaxed_model.addConstr(var <= value if sense == '<=' else var >= value)

        relaxed_model.optimize()

        if relaxed_model.Status == GRB.OPTIMAL:
            result['status'] = 'OPTIMAL'
            result['objective'] = relaxed_model.ObjVal
            result['solution'] = {v.VarName: v.X for v in relaxed_model.getVars()}
            result['vbasis'] = {v.VarName: v.VBasis for v in relaxed_model.getVars()}
            result['cbasis'] = {c.ConstrName: c.CBasis for c in relaxed_model.getConstrs()}
        elif relaxed_model.Status == GRB.INFEASIBLE:
            result['status'] = 'INFEASIBLE'
        elif relaxed_model.Status == GRB.UNBOUNDED:
            result['status'] = 'UNBOUNDED'

    except gp.GurobiError as e:
        logger.error(f"A Gurobi error occurred during LP relaxation solve: {e}")
        result['status'] = 'ERROR'
    
    finally:
        # Ensure the temporary model is always disposed of
        if relaxed_model:
            relaxed_model.dispose()

    return result