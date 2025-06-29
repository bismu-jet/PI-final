# solver/gurobi_interface.py

import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
from solver.problem import MIPProblem

def solve_lp_relaxation(problem: MIPProblem, local_constraints: List[Tuple[str, str, float]], time_limit: Optional[float]) -> Dict:
    """
    Solves the LP relaxation. This version RELAXES INTEGER VARIABLES to
    ensure Gurobi acts as a pure LP solver.
    """
    model_copy = None
    try:
        model_copy = problem.model.copy()

        # --- THIS IS THE MOST IMPORTANT STEP IN THE ENTIRE FILE ---
        # We iterate through all variables in our copied model.
        # If a variable is an Integer or Binary, we change its type to Continuous.
        # This prevents Gurobi's MIP solver from activating.
        for var in model_copy.getVars():
            if var.VType in [GRB.BINARY, GRB.INTEGER]:
                var.VType = GRB.CONTINUOUS
        # -----------------------------------------------------------------

        for var_name, sense, bound in local_constraints:
            var = model_copy.getVarByName(var_name)
            if sense == '<=': model_copy.addConstr(var <= bound)
            else: model_copy.addConstr(var >= bound)

        model_copy.setParam('Presolve', 0)
        model_copy.setParam('Cuts', 0)
        model_copy.setParam('Heuristics', 0)
        model_copy.setParam('LogToConsole', 0)

        if time_limit is not None and time_limit > 0:
            model_copy.setParam(GRB.Param.TimeLimit, time_limit)

        model_copy.optimize()

        # Process and return results... (rest of the function is the same)
        if model_copy.status == GRB.Status.OPTIMAL:
            return {'status': 'OPTIMAL', 'objective': model_copy.ObjVal, 'solution': {v.VarName: v.X for v in model_copy.getVars()}}
        elif model_copy.status == GRB.Status.TIME_LIMIT:
             return {'status': 'TIME_LIMIT', 'objective': model_copy.ObjVal if model_copy.SolCount > 0 else None, 'solution': {v.VarName: v.X for v in model_copy.getVars()} if model_copy.SolCount > 0 else None}
        else:
            return {'status': model_copy.status, 'objective': None, 'solution': None}

    except gp.GurobiError as e:
        return {'status': f'GUROBI_ERROR: {e}', 'objective': None, 'solution': None}
    finally:
        if model_copy:
            model_copy.dispose()