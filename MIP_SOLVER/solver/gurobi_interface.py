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
    Solves the LP relaxation of the MIP problem with its original objective.
    """
    result = {'status': 'UNKNOWN'}
    relaxed_model = None

    try:
        relaxed_model = problem.model.relax()
        
        for var_name, sense, value in local_constraints:
            var = relaxed_model.getVarByName(var_name)
            if sense == '<=': relaxed_model.addConstr(var <= value)
            else: relaxed_model.addConstr(var >= value)
        
        if cuts:
            for i, cut in enumerate(cuts):
                cut_coeffs_dict = cut['coeffs']
                cut_rhs = cut['rhs']
                cut_sense = cut['sense']
                
                new_expr = gp.LinExpr()
                for var_name, coeff in cut_coeffs_dict.items():
                    var = relaxed_model.getVarByName(var_name)
                    if var is not None:
                        new_expr.add(var, coeff)
                
                if cut_sense == GRB.GREATER_EQUAL:
                    relaxed_model.addConstr(new_expr >= cut_rhs, name=f"cut_{i}")
                else:
                    relaxed_model.addConstr(new_expr <= cut_rhs, name=f"cut_{i}")

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

def solve_lp_with_custom_objective(problem: MIPProblem, 
                                   objective_coeffs: Dict[str, float]) -> Dict[str, Any]:
    """
    Solves an LP with a custom objective function, built from a dictionary.
    """
    result = {'status': 'UNKNOWN'}
    model_copy = None
    try:
        # This creates a new, local model instance.
        model_copy = problem.model.relax()
        
        # --- THIS IS THE FIX ---
        # 1. The function now accepts a dictionary `objective_coeffs`.
        # 2. It builds the objective expression locally, using variables
        #    that belong to its own `model_copy`. This prevents the
        #    "Variable not in model" error.
        objective_expr = gp.LinExpr()
        for var_name, coeff in objective_coeffs.items():
            # Get the variable object FROM THE LOCAL MODEL COPY.
            var = model_copy.getVarByName(var_name)
            if var is not None:
                objective_expr.add(var, coeff)
        # --- END FIX ---
        
        model_copy.setObjective(objective_expr, GRB.MINIMIZE)
        
        model_copy.optimize()

        if model_copy.Status == GRB.OPTIMAL:
            result['status'] = 'OPTIMAL'
            result['objective'] = model_copy.ObjVal
            result['solution'] = {v.VarName: v.X for v in model_copy.getVars()}
        else:
            result['status'] = 'INFEASIBLE'

    except gp.GurobiError as e:
        logger.error(f"A Gurobi error occurred during custom objective LP solve: {e}")
        result['status'] = 'ERROR'
    
    finally:
        if model_copy:
            model_copy.dispose()
            
    return result