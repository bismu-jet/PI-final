import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Any, Optional

from solver.problem import MIPProblem
from solver.utilities import setup_logger

logger = setup_logger()

def solve_lp_relaxation(problem: MIPProblem, 
                        local_constraints: List[Tuple[str, str, float]], 
                        cuts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Solves the LP relaxation of the MIP problem.
    It can optionally include a list of generated cuts.
    """
    result = {'status': 'UNKNOWN'}
    relaxed_model = None

    try:
        # The .relax() method will create a new model that inherits the parent's environment and parameters.
        relaxed_model = problem.model.relax()

        # Apply local branching constraints
        for var_name, sense, value in local_constraints:
            var = relaxed_model.getVarByName(var_name)
            if sense == '<=':
                relaxed_model.addConstr(var <= value)
            elif sense == '>=':
                relaxed_model.addConstr(var >= value)
        
        # --- CORRECTED: REBUILD CUT EXPRESSION FROM DATA ---
        if cuts:
            for i, cut in enumerate(cuts):
                # Expects a dictionary like: {'coeffs': {'x1': 0.5, ...}, 'sense': ..., 'rhs': ...}
                cut_coeffs_dict = cut['coeffs']
                cut_rhs = cut['rhs']
                cut_sense = cut['sense']
                
                # Create a new, empty expression within the current model's context
                new_expr = gp.LinExpr()
                for var_name, coeff in cut_coeffs_dict.items():
                    # Get the variable object that belongs to THIS model
                    var = relaxed_model.getVarByName(var_name)
                    if var is not None:
                        new_expr.add(var, coeff)
                
                # Add the newly constructed constraint
                if cut_sense == GRB.GREATER_EQUAL:
                    relaxed_model.addConstr(new_expr >= cut_rhs, name=f"gomory_{i}")
                elif cut_sense == GRB.LESS_EQUAL:
                    relaxed_model.addConstr(new_expr <= cut_rhs, name=f"gomory_{i}")
                else: # GRB.EQUAL
                    relaxed_model.addConstr(new_expr == cut_rhs, name=f"gomory_{i}")
        # --- END CORRECTION ---

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
        if relaxed_model:
            relaxed_model.dispose()

    return result
