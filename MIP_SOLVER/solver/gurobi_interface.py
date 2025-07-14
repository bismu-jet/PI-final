import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Any, Optional

# --- Importing custom modules for the solver ---
from solver.problem import MIPProblem # To load and manage the MIP problem data.
from solver.utilities import setup_logger # Utility for logging solver progress.

# --- Setup the logger ---
# Creates a logger object to print progress and debug information for Gurobi interactions.
logger = setup_logger()

def solve_lp_relaxation(problem: MIPProblem, 
                        local_constraints: List[Tuple[str, str, float]], 
                        cuts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Solves the Linear Programming (LP) relaxation of the MIP problem for a specific
    node in the Branch and Bound tree. An LP relaxation is created by dropping the
    integrality requirement of the integer variables.

    This function also applies any local branching constraints and cutting planes
    relevant to the current node.

    Args:
        problem (MIPProblem): The original MIP problem instance.
        local_constraints (List[Tuple[str, str, float]]): Branching constraints
            (e.g., [('x1', '<=', 0)]) defining the current B&B node.
        cuts (Optional[List[Dict[str, Any]]]): A list of cutting planes to add to
            the LP relaxation to tighten it.

    Returns:
        Dict[str, Any]: A dictionary containing the solution status, objective value,
        variable values, basis information (for warm-starting), and the Gurobi
        model object itself. **The caller is responsible for calling .dispose() on the model**.
    """
    # Initialize the result dictionary.
    result: Dict[str, Any] = {'status': 'UNKNOWN', 'model': None}
    relaxed_model: Optional[gp.Model] = None

    try:
        # Create the LP relaxation from the original MIP model.
        relaxed_model = problem.model.relax()
        
        # Disable Gurobi's own presolve and dual reductions to have full control.
        # This is crucial for a custom solver to ensure the model isn't altered
        # in unexpected ways.
        relaxed_model.setParam(GRB.Param.Presolve, 0)
        relaxed_model.setParam(GRB.Param.DualReductions, 0) 
        
        # Apply the local branching constraints that define the current node.
        for var_name, sense, value in local_constraints:
            var: gp.Var = relaxed_model.getVarByName(var_name)
            if var is not None:
                if sense == '<=': relaxed_model.addConstr(var <= value)
                else: relaxed_model.addConstr(var >= value)
        
        # If any cutting planes have been generated, add them to the model.
        if cuts:
            for i, cut in enumerate(cuts):
                cut_coeffs_dict, cut_rhs, cut_sense = cut['coeffs'], cut['rhs'], cut['sense']
                # Build the linear expression for the cut.
                expr: gp.LinExpr = gp.LinExpr([(relaxed_model.getVarByName(v), c) for v, c in cut_coeffs_dict.items()])
                relaxed_model.addConstr(expr, sense=cut_sense, rhs=cut_rhs, name=f"cut_{i}")

        # --- Solve the model ---
        relaxed_model.optimize()

        # --- Process the result ---
        if relaxed_model.Status == GRB.OPTIMAL:
            result['status'] = 'OPTIMAL'
            result['objective'] = relaxed_model.ObjVal
            # Store the solution (variable values).
            result['solution'] = {v.VarName: v.X for v in relaxed_model.getVars()}
            # Store the basis information, which can be used to "warm start"
            # subsequent LP solves, making them faster.
            result['vbasis'] = {v.VarName: v.VBasis for v in relaxed_model.getVars()}
            result['cbasis'] = {c.ConstrName: c.CBasis for c in relaxed_model.getConstrs()}
        else:
            # The LP could be infeasible or unbounded.
            result['status'] = 'INFEASIBLE' if relaxed_model.Status == GRB.INFEASIBLE else 'UNBOUNDED'

    except gp.GurobiError as e:
        logger.error(f"An error occurred in Gurobi: {e}", exc_info=True)
        result['status'] = 'ERROR'
        # If an error occurs, we must dispose of the model to prevent memory leaks.
        if relaxed_model:
            relaxed_model.dispose() 
    
    # Pass the model object back to the caller. This is unconventional but allows
    # the caller to extract more complex information if needed, but it MUST be disposed of later.
    result['model'] = relaxed_model
    return result

def solve_lp_with_custom_objective(problem: MIPProblem,
                                     objective_coeffs: Dict[str, float]) -> Dict[str, Any]:
    """
    Solves an LP relaxation of the problem but with a custom objective function.
    This is commonly used in heuristics like a Feasibility Pump, where the goal
    is not to optimize the original objective but to minimize the distance to
    an integer-infeasible solution to find a nearby integer-feasible one.

    Args:
        problem (MIPProblem): The original MIP problem instance.
        objective_coeffs (Dict[str, float]): A dictionary mapping variable names
            to their coefficients in the new objective function.

    Returns:
        Dict[str, Any]: A dictionary containing the solution status, objective value,
        and variable values.
    """
    result: Dict[str, Any] = {'status': 'UNKNOWN'}
    model_copy: Optional[gp.Model] = None
    try:
        # Create a relaxed copy of the model.
        model_copy = problem.model.relax()
        
        # Build the new objective function as a linear expression.
        objective_expr: gp.LinExpr = gp.LinExpr()
        for var_name, coeff in objective_coeffs.items():
            var: gp.Var = model_copy.getVarByName(var_name)
            if var is not None:
                objective_expr.add(var, coeff)
        
        # Set the new objective in the model (typically for minimization in heuristics).
        model_copy.setObjective(objective_expr, GRB.MINIMIZE)
        
        # Solve the LP with the new objective.
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
        # Always dispose of the model copy to free up Gurobi resources.
        if model_copy:
            model_copy.dispose()
            
    return result

def solve_sub_mip(problem: MIPProblem,
                  fixed_vars: Dict[str, float],
                  time_limit: float) -> Dict[str, Any]:
    """
    Solves a smaller sub-MIP based on the original problem. This is a common
    heuristic technique (like RINS or Local Branching) where some variables
    are fixed to specific values to explore a limited "neighborhood" of the
    solution space for a new incumbent.

    Args:
        problem (MIPProblem): The original MIP problem instance.
        fixed_vars (Dict[str, float]): A dictionary of variables to fix and their values.
        time_limit (float): A time limit for the sub-MIP solve.

    Returns:
        Dict[str, Any]: A dictionary with the status and, if found, the solution.
    """
    result: Dict[str, Any] = {'status': 'UNKNOWN'}
    sub_mip_model: Optional[gp.Model] = None
    try:
        # Create a full copy of the original MIP model, not a relaxation.
        sub_mip_model = problem.model.copy()
        
        # Disable Gurobi's advanced features to run a more "pure" or controlled search.
        sub_mip_model.setParam('Presolve', 0)
        sub_mip_model.setParam('Cuts', 0)
        sub_mip_model.setParam('Heuristics', 0)
        sub_mip_model.setParam('TimeLimit', time_limit)
        
        # Fix the specified variables by setting their lower and upper bounds to the same value.
        for var_name, value in fixed_vars.items():
            var: gp.Var = sub_mip_model.getVarByName(var_name)
            if var is not None:
                var.LB, var.UB = value, value
                
        # Solve the sub-MIP.
        sub_mip_model.optimize()
        
        # Check if Gurobi found at least one feasible solution.
        if sub_mip_model.SolCount > 0:
            result['status'] = 'FEASIBLE'
            result['objective'] = sub_mip_model.ObjVal
            result['solution'] = {v.VarName: v.X for v in sub_mip_model.getVars()}
        else:
            result['status'] = 'NO_SOLUTION_FOUND'
            
    except gp.GurobiError as e:
        logger.error(f"A Gurobi error occurred during sub-MIP solve: {e}")
        result['status'] = 'ERROR'
        
    finally:
        # Always dispose of the model copy.
        if sub_mip_model:
            sub_mip_model.dispose()
            
    return result