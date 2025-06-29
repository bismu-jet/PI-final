import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Any
from solver.problem import MIPProblem

def solve_lp_relaxation(problem: MIPProblem, local_constraints: List[Tuple[str, str, float]]) -> Dict[str, Any]:
    """
    Solves the LP relaxation of a given MIP problem with additional local constraints.

    Args:
        problem (MIPProblem): The MIPProblem instance containing the base Gurobi model.
        local_constraints (List[Tuple[str, str, float]]): A list of constraints to apply
                                                        to the current LP relaxation.
                                                        E.g., [( 'x1 ',  '<= ', 0), ( 'x2 ',  '>= ', 1)].

    Returns:
        Dict[str, Any]: A dictionary containing the status, objective value, and solution
                        of the LP relaxation. Format: {
                             'status': str,  # OPTIMAL, INFEASIBLE, UNBOUNDED, etc.
                             'objective ': float | None,
                             'solution ': Dict[str, float] | None
                        }
    """
    # Create a copy of the base Gurobi model to avoid modifying the original
    model = problem.model.copy()
    model.setParam("OutputFlag", 1)  # Suppress Gurobi output

    for var in model.getVars():
        if var.VType in [GRB.BINARY, GRB.INTEGER]:
            var.VType = GRB.CONTINUOUS
    # --- END OF THE NEW CODE BLOCK ---

    # Apply local constraints
    for var_name, sense, rhs in local_constraints:
        var = model.getVarByName(var_name)
        if var is None:
            raise ValueError(f"Variable {var_name} not found in model.")
        if sense == ">=":
            model.addConstr(var >= rhs, name=f"local_con_{var_name}_ge_{rhs}")
        elif sense == "<=":
            model.addConstr(var <= rhs, name=f"local_con_{var_name}_le_{rhs}")
        elif sense == "==":
            model.addConstr(var == rhs, name=f"local_con_{var_name}_eq_{rhs}")
        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")

    # Set Gurobi parameters to ensure it only solves the LP relaxation
    model.setParam('Presolve', 0)
    model.setParam('Cuts', 0)
    model.setParam('Heuristics', 0)
    
    # --- ADD THIS LINE FOR DEBUGGING ---
    model.setParam('LogToConsole', 1) 
    # -----------------------------------
    try:

        model.optimize()

        if model.status== GRB.OPTIMAL:
            return {
                 'status':  'OPTIMAL',
                 'objective': model.ObjVal,
                 'solution': {v.VarName: v.X for v in model.getVars()}
            }
        elif model.status== GRB.INFEASIBLE:
            return {
                 'status':  'INFEASIBLE',
                 'objective': None,
                 'solution': None
            }
        elif model.status== GRB.UNBOUNDED:
            return {
                 'status':  'UNBOUNDED',
                 'objective': None,
                 'solution': None
            }
        else:
            # Handle other statuses gracefully, e.g., TIME_LIMIT, INF_OR_UNBD
            return {
                 'status': GRB.Status[model.status],
                 'objective': None,
                 'solution': None
            }
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        return {
             'status':  'ERROR',
             'objective': None,
             'solution': None
        }
    finally:
        model.dispose() # Release Gurobi resources
